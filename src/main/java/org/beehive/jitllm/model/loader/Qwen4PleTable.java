package org.beehive.jitllm.model.loader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.PleHash;

/**
 * Direct, bounded host staging from the original NVMe checkpoint; no table copy is needed.
 *
 * <p>Row data is read through a table reader that may be dedicated to PLE (see
 * {@link #PLE_PAGE_CACHE_PROPERTY}) so PLE rows and dense weights stop thrashing one shared
 * page cache. The legacy constructor keeps the exact shared-reader behavior.
 */
public final class Qwen4PleTable implements AutoCloseable {
    /** System property sizing the dedicated PLE page cache, in bytes. */
    public static final String PLE_PAGE_CACHE_PROPERTY = "jitllm.qwen4.plePageCacheBytes";
    /** Dedicated PLE page-cache budget used when the property is unset. */
    public static final long DEFAULT_PLE_PAGE_CACHE_BYTES = 256L << 20;
    /** Property value selecting the legacy mode: PLE rows share the caller's reader. */
    public static final long SHARED_PLE_PAGE_CACHE_BYTES = 0L;

    /** Resolve the PLE page-cache budget: {@link #SHARED_PLE_PAGE_CACHE_BYTES} means shared. */
    public static long plePageCacheBytes() {
        long bytes = Long.getLong(PLE_PAGE_CACHE_PROPERTY, DEFAULT_PLE_PAGE_CACHE_BYTES);
        if (bytes < 0) throw new IllegalArgumentException(PLE_PAGE_CACHE_PROPERTY + " must be >= 0: " + bytes);
        return bytes;
    }

    private final PagedTensorReader reader;
    private final boolean ownsReader;
    private final List<SafeTensorIndex.Tensor> shards;
    private final long[] ends;
    private final int width, elementBytes;
    private final float scale;
    private final PleHash hash;
    private boolean closed;

    /** Legacy shared mode: hash buffers, scale, and rows all read through {@code reader}. */
    public Qwen4PleTable(SafeTensorIndex index, PagedTensorReader reader, String prefix, int eos) throws IOException {
        this(index, reader, reader, false, prefix, eos);
    }

    /** Rows read through {@code tableReader}; small hash/scale tensors through {@code metadataReader}. Neither is closed here. */
    public Qwen4PleTable(SafeTensorIndex index, PagedTensorReader metadataReader, PagedTensorReader tableReader,
                          String prefix, int eos) throws IOException {
        this(index, metadataReader, tableReader, false, prefix, eos);
    }

    /** Dedicated mode: rows read through a new reader of {@code plePageCacheBytes}, owned and closed by this table. */
    public Qwen4PleTable(SafeTensorIndex index, PagedTensorReader metadataReader, long plePageCacheBytes,
                          String prefix, int eos) throws IOException {
        this(index, metadataReader, new PagedTensorReader(plePageCacheBytes, metadataReader.pageBytes()), true, prefix, eos);
    }

    private Qwen4PleTable(SafeTensorIndex index, PagedTensorReader metadataReader, PagedTensorReader tableReader,
                          boolean ownsReader, String prefix, int eos) throws IOException {
        this.reader = tableReader;
        this.ownsReader = ownsReader;
        this.hash = new PleHash(metadataReader.readLongs(index.require(prefix + ".layer_multipliers")),
                metadataReader.readLongs(index.require(prefix + ".ngram_heads_vocab_sizes")),
                metadataReader.readLongs(index.require(prefix + ".ngram_heads_offsets")), eos);
        String tablePrefix = prefix + ".ngram_embedding";
        TreeMap<Integer, SafeTensorIndex.Tensor> parts = new TreeMap<>();
        for (var tensor : index.tensors()) {
            if (tensor.name().startsWith(tablePrefix + ".shard_") && tensor.name().endsWith(".weight")) {
                String number = tensor.name().substring((tablePrefix + ".shard_").length(), tensor.name().length() - ".weight".length());
                parts.put(Integer.parseInt(number), tensor);
            }
        }
        if (parts.isEmpty()) parts.put(0, index.require(tablePrefix + ".weight"));
        for (int i = 0; i < parts.size(); i++) if (!parts.containsKey(i)) throw new IOException("Missing PLE shard " + i);
        shards = List.copyOf(parts.values());
        var first = shards.get(0);
        if (first.shape().size() != 2) throw new IOException("PLE table must be a matrix");
        width = Math.toIntExact(first.shape().get(1));
        elementBytes = switch (first.dtype()) { case "F8_E4M3" -> 1; case "BF16" -> 2; default -> throw new IOException("Unsupported PLE dtype " + first.dtype()); };
        ends = new long[shards.size()];
        long end = 0;
        for (int i = 0; i < shards.size(); i++) {
            var tensor = shards.get(i);
            if (tensor.shape().size() != 2 || tensor.shape().get(1) != width || !tensor.dtype().equals(first.dtype()))
                throw new IOException("PLE shard layout mismatch");
            ends[i] = end = Math.addExact(end, tensor.shape().get(0));
        }
        long[] offsets = metadataReader.readLongs(index.require(prefix + ".ngram_heads_offsets"));
        long[] sizes = metadataReader.readLongs(index.require(prefix + ".ngram_heads_vocab_sizes"));
        for (int i = 0; i < offsets.length; i++) if (Math.addExact(offsets[i], sizes[i]) > end) throw new IOException("PLE heads exceed table rows");
        var scalar = index.require(tablePrefix + ".weight_scale");
        if (scalar.elements() != 1) throw new IOException("PLE scale must be scalar");
        ByteBuffer bytes = ByteBuffer.wrap(metadataReader.read(scalar, 0, Math.toIntExact(scalar.bytes()))).order(ByteOrder.LITTLE_ENDIAN);
        scale = switch (scalar.dtype()) { case "BF16" -> FloatFormats.bf16(bytes.getShort()); case "F32" -> bytes.getFloat(); default -> throw new IOException("Unsupported PLE scale dtype"); };
        if (!Float.isFinite(scale) || scale <= 0) throw new IOException("Invalid PLE scale");
    }

    public PleHash hash() { return hash; }
    public int width() { return width; }
    public long rows() { return ends[ends.length - 1]; }
    public float scale() { return scale; }
    /** The reader row data is paged through (dedicated to PLE unless shared mode was selected). */
    public PagedTensorReader reader() { return reader; }
    /** Page-cache statistics of the table reader, in {@link PagedTensorReader} style. */
    public PagedTensorReader.Statistics statistics() { return reader.statistics(); }

    private int shardFor(long row) {
        int shard = Arrays.binarySearch(ends, row + 1);
        return shard < 0 ? -shard - 1 : shard;
    }

    private long shardBegin(int shard) { return shard == 0 ? 0 : ends[shard - 1]; }

    private void decodeRow(byte[] bytes, float[] destination, int offset) {
        for (int column = 0; column < width; column++) {
            float value = elementBytes == 1 ? FloatFormats.e4m3(bytes[column])
                    : FloatFormats.bf16((bytes[column * 2] & 255) | ((bytes[column * 2 + 1] & 255) << 8));
            // SGLang casts gathered FP8 values to BF16 before applying the BF16 scale.
            float scaled = value * scale;
            int bits = Float.floatToRawIntBits(scaled);
            bits += 0x7fff + ((bits >>> 16) & 1);
            destination[offset + column] = FloatFormats.bf16(bits >>> 16);
        }
    }

    private void validate(long[] rows) {
        for (long row : rows) if (row < 0 || row >= rows()) throw new IndexOutOfBoundsException("PLE row " + row);
    }

    /** Per-row reference gather in request order. Prefer {@link #fetchRows} for batches. */
    public void gather(long[] rows, float[] destination) throws IOException {
        if (destination.length != Math.multiplyExact(rows.length, width)) throw new IllegalArgumentException("Wrong staging length");
        // Validate the whole batch before touching output.
        validate(rows);
        byte[] bytes = new byte[Math.multiplyExact(width, elementBytes)];
        for (int i = 0; i < rows.length; i++) {
            long row = rows[i];
            int shard = shardFor(row);
            reader.read(shards.get(shard), Math.multiplyExact(row - shardBegin(shard), bytes.length), bytes, 0, bytes.length);
            decodeRow(bytes, destination, i * width);
        }
    }

    /**
     * Batched gather bit-identical to {@link #gather}: repeated rows are decoded once, unique
     * rows are visited in ascending row order (hence ascending shard offset), so each needed
     * page is filled at most once, and results are scattered back in request order.
     */
    public void fetchRows(long[] rowIds, float[] destination) throws IOException {
        if (destination.length != Math.multiplyExact(rowIds.length, width)) throw new IllegalArgumentException("Wrong staging length");
        validate(rowIds);
        Integer[] order = new Integer[rowIds.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingLong(i -> rowIds[i]));
        byte[] bytes = new byte[Math.multiplyExact(width, elementBytes)];
        float[] decoded = new float[width];
        for (int start = 0; start < order.length; ) {
            long row = rowIds[order[start]];
            int shard = shardFor(row);
            reader.read(shards.get(shard), Math.multiplyExact(row - shardBegin(shard), bytes.length), bytes, 0, bytes.length);
            decodeRow(bytes, decoded, 0);
            int end = start;
            while (end < order.length && rowIds[order[end]] == row) {
                System.arraycopy(decoded, 0, destination, order[end] * width, width);
                end++;
            }
            start = end;
        }
    }

    public float[] fetchRows(long[] rowIds) throws IOException {
        float[] destination = new float[Math.multiplyExact(rowIds.length, width)];
        fetchRows(rowIds, destination);
        return destination;
    }

    public void fetchRows(int[] rowIds, float[] destination) throws IOException {
        fetchRows(toLongs(rowIds), destination);
    }

    public float[] fetchRows(int[] rowIds) throws IOException {
        return fetchRows(toLongs(rowIds));
    }

    /** Warm the pages covering {@code rowIds} (deduped, ascending, contiguous runs coalesced) without dequantizing. */
    public void prefetchRows(long[] rowIds) throws IOException {
        validate(rowIds);
        long[] sorted = rowIds.clone();
        Arrays.sort(sorted);
        int rowBytes = Math.multiplyExact(width, elementBytes);
        for (int start = 0; start < sorted.length; ) {
            long first = sorted[start];
            int shard = shardFor(first);
            int end = start + 1;
            while (end < sorted.length && (sorted[end] == sorted[end - 1]
                    || (sorted[end] == sorted[end - 1] + 1 && shardFor(sorted[end]) == shard))) end++;
            long rows = sorted[end - 1] - first + 1;
            reader.prefetch(shards.get(shard), Math.multiplyExact(first - shardBegin(shard), rowBytes),
                    Math.multiplyExact(rows, rowBytes));
            start = end;
        }
    }

    public void prefetchRows(int[] rowIds) throws IOException {
        prefetchRows(toLongs(rowIds));
    }

    private static long[] toLongs(int[] rowIds) {
        long[] rows = new long[rowIds.length];
        for (int i = 0; i < rowIds.length; i++) rows[i] = rowIds[i];
        return rows;
    }

    /** Close the table reader only when this table created it (dedicated mode). */
    @Override public void close() throws IOException {
        if (closed) return;
        closed = true;
        if (ownsReader) reader.close();
    }
}

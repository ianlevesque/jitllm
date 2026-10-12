package org.beehive.jitllm.model.loader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.PleHash;

/** Direct, bounded host staging from the original NVMe checkpoint; no table copy is needed. */
public final class Qwen4PleTable {
    private final PagedTensorReader reader;
    private final List<SafeTensorIndex.Tensor> shards;
    private final long[] ends;
    private final int width, elementBytes;
    private final float scale;
    private final PleHash hash;

    public Qwen4PleTable(SafeTensorIndex index, PagedTensorReader reader, String prefix, int eos) throws IOException {
        this.reader = reader;
        this.hash = new PleHash(reader.readLongs(index.require(prefix + ".layer_multipliers")),
                reader.readLongs(index.require(prefix + ".ngram_heads_vocab_sizes")),
                reader.readLongs(index.require(prefix + ".ngram_heads_offsets")), eos);
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
        long[] offsets = reader.readLongs(index.require(prefix + ".ngram_heads_offsets"));
        long[] sizes = reader.readLongs(index.require(prefix + ".ngram_heads_vocab_sizes"));
        for (int i = 0; i < offsets.length; i++) if (Math.addExact(offsets[i], sizes[i]) > end) throw new IOException("PLE heads exceed table rows");
        var scalar = index.require(tablePrefix + ".weight_scale");
        if (scalar.elements() != 1) throw new IOException("PLE scale must be scalar");
        ByteBuffer bytes = ByteBuffer.wrap(reader.read(scalar, 0, Math.toIntExact(scalar.bytes()))).order(ByteOrder.LITTLE_ENDIAN);
        scale = switch (scalar.dtype()) { case "BF16" -> FloatFormats.bf16(bytes.getShort()); case "F32" -> bytes.getFloat(); default -> throw new IOException("Unsupported PLE scale dtype"); };
        if (!Float.isFinite(scale) || scale <= 0) throw new IOException("Invalid PLE scale");
    }

    public PleHash hash() { return hash; }
    public int width() { return width; }
    public long rows() { return ends[ends.length - 1]; }
    public float scale() { return scale; }

    public void gather(long[] rows, float[] destination) throws IOException {
        if (destination.length != Math.multiplyExact(rows.length, width)) throw new IllegalArgumentException("Wrong staging length");
        // Validate the whole batch before touching output.
        for (long row : rows) if (row < 0 || row >= rows()) throw new IndexOutOfBoundsException("PLE row " + row);
        byte[] bytes = new byte[Math.multiplyExact(width, elementBytes)];
        for (int i = 0; i < rows.length; i++) {
            long row = rows[i];
            int shard = Arrays.binarySearch(ends, row + 1);
            if (shard < 0) shard = -shard - 1;
            long begin = shard == 0 ? 0 : ends[shard - 1];
            reader.read(shards.get(shard), Math.multiplyExact(row - begin, bytes.length), bytes, 0, bytes.length);
            for (int column = 0; column < width; column++) {
                float value = elementBytes == 1 ? FloatFormats.e4m3(bytes[column])
                        : FloatFormats.bf16((bytes[column * 2] & 255) | ((bytes[column * 2 + 1] & 255) << 8));
                // SGLang casts gathered FP8 values to BF16 before applying the BF16 scale.
                float scaled = value * scale;
                int bits = Float.floatToRawIntBits(scaled);
                bits += 0x7fff + ((bits >>> 16) & 1);
                destination[i * width + column] = FloatFormats.bf16(bits >>> 16);
            }
        }
    }
}

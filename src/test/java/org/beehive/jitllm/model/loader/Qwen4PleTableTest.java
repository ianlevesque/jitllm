package org.beehive.jitllm.model.loader;

import java.io.ByteArrayOutputStream;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.junit.Test;
import static org.junit.Assert.*;

public final class Qwen4PleTableTest {
    private static final String PREFIX = "test.ple_embedding";
    private static final int EOS = 7;
    private static final int ROWS = 12, WIDTH = 4;

    /** Two-shard synthetic checkpoint, mirroring Qwen4LayerStateTest's writer. */
    private static final class Fixture implements AutoCloseable {
        final Path directory = Files.createTempDirectory("qwen4-ple-table-");
        final Map<String, Object> header = new LinkedHashMap<>();
        final ByteArrayOutputStream data = new ByteArrayOutputStream();
        final SafeTensorIndex index;

        Fixture(String dtype) throws Exception {
            longs(PREFIX + ".layer_multipliers", 3, 5);
            longs(PREFIX + ".ngram_heads_vocab_sizes", ROWS);
            longs(PREFIX + ".ngram_heads_offsets", 0);
            var scale = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
            scale.putShort((short) (Float.floatToRawIntBits(1.5f) >>> 16));
            add(PREFIX + ".ngram_embedding.weight_scale", "BF16", List.of(1, 1), scale.array());
            for (int shard = 0; shard < 2; shard++) {
                byte[] bytes = dtype.equals("F8_E4M3") ? fp8Shard(shard) : bf16Shard(shard);
                add(PREFIX + ".ngram_embedding.shard_" + shard + ".weight", dtype, List.of(ROWS / 2, WIDTH), bytes);
            }
            byte[] metadata = MetadataJson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            var file = ByteBuffer.allocate(8 + metadata.length + data.size()).order(ByteOrder.LITTLE_ENDIAN);
            file.putLong(metadata.length).put(metadata).put(data.toByteArray());
            Files.write(directory.resolve("model.safetensors"), file.array());
            index = SafeTensorIndex.open(directory);
        }

        private static byte[] fp8Shard(int shard) {
            byte[] bytes = new byte[ROWS / 2 * WIDTH];
            for (int i = 0; i < bytes.length; i++) {
                int row = shard * ROWS / 2 + i / WIDTH, column = i % WIDTH;
                bytes[i] = (byte) (0x20 + (row * 7 + column * 13) % 0x5f);
            }
            if (shard == 0) {
                bytes[0] = 0; bytes[1] = (byte) 0x80; bytes[2] = 0x7f; // zero, -0, NaN
                bytes[3] = (byte) 0xfe; bytes[4] = 0x01; // -max, smallest subnormal
            }
            return bytes;
        }

        private static byte[] bf16Shard(int shard) {
            var bytes = ByteBuffer.allocate(ROWS / 2 * WIDTH * 2).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < ROWS / 2 * WIDTH; i++) {
                int row = shard * ROWS / 2 + i / WIDTH, column = i % WIDTH;
                bytes.putShort((short) (0x3f80 + row * 37 + column * 11));
            }
            return bytes.array();
        }

        void add(String name, String dtype, List<Integer> shape, byte[] bytes) throws Exception {
            int start = data.size();
            data.write(bytes);
            header.put(name, Map.of("dtype", dtype, "shape", shape, "data_offsets", List.of(start, data.size())));
        }

        void longs(String name, long... values) throws Exception {
            var bytes = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (long value : values) bytes.putLong(value);
            add(name, "I64", List.of(values.length), bytes.array());
        }

        Qwen4PleTable table(PagedTensorReader reader) throws Exception {
            return new Qwen4PleTable(index, reader, PREFIX, EOS);
        }

        @Override public void close() throws Exception {
            try (var files = Files.walk(directory)) {
                for (var file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
            }
        }
    }

    private static float[] perRowReference(Qwen4PleTable table, long[] rows) throws Exception {
        float[] expected = new float[rows.length * table.width()];
        float[] single = new float[table.width()];
        for (int i = 0; i < rows.length; i++) {
            table.gather(new long[]{rows[i]}, single);
            System.arraycopy(single, 0, expected, i * table.width(), single.length);
        }
        return expected;
    }

    private static void assertBitEqual(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++)
            assertEquals("bits at " + i, Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(actual[i]));
    }

    @Test public void batchedFetchIsBitIdenticalToPerRowGatherFp8() throws Exception {
        try (var fixture = new Fixture("F8_E4M3"); var reader = new PagedTensorReader(4096, 16)) {
            var table = fixture.table(reader);
            long[] rows = {11, 0, 5, 5, 6, 11, 3, 0, 1, 2, 4, 6};
            float[] expected = perRowReference(table, rows);
            assertBitEqual(expected, table.fetchRows(rows));
            float[] into = new float[rows.length * WIDTH];
            table.fetchRows(rows, into);
            assertBitEqual(expected, into);
            float[] gathered = new float[rows.length * WIDTH];
            table.gather(rows, gathered);
            assertBitEqual(expected, gathered);
        }
    }

    @Test public void batchedFetchIsBitIdenticalToPerRowGatherBf16() throws Exception {
        try (var fixture = new Fixture("BF16"); var reader = new PagedTensorReader(4096, 16)) {
            var table = fixture.table(reader);
            long[] rows = {10, 2, 2, 7, 0, 11, 7};
            assertBitEqual(perRowReference(table, rows), table.fetchRows(rows));
        }
    }

    @Test public void intOverloadMatchesLongOverload() throws Exception {
        try (var fixture = new Fixture("F8_E4M3"); var reader = new PagedTensorReader(4096, 16)) {
            var table = fixture.table(reader);
            assertBitEqual(table.fetchRows(new long[]{4, 9, 4}), table.fetchRows(new int[]{4, 9, 4}));
        }
    }

    @Test public void duplicateRowsDoNotReReadPages() throws Exception {
        try (var fixture = new Fixture("F8_E4M3");
             var duplicates = new PagedTensorReader(4096, 16);
             var unique = new PagedTensorReader(4096, 16)) {
            var duplicateTable = fixture.table(duplicates);
            var uniqueTable = fixture.table(unique);
            var beforeDuplicates = duplicates.statistics();
            var beforeUnique = unique.statistics();
            duplicateTable.fetchRows(new long[]{5, 5, 5, 11, 5, 11});
            uniqueTable.fetchRows(new long[]{5, 11});
            var duplicateStats = duplicates.statistics();
            var uniqueStats = unique.statistics();
            assertEquals(uniqueStats.misses() - beforeUnique.misses(), duplicateStats.misses() - beforeDuplicates.misses());
            assertEquals(uniqueStats.bytesRead() - beforeUnique.bytesRead(), duplicateStats.bytesRead() - beforeDuplicates.bytesRead());
            assertEquals(uniqueStats.hits() - beforeUnique.hits(), duplicateStats.hits() - beforeDuplicates.hits());
        }
    }

    @Test public void requestOrderDoesNotReReadPages() throws Exception {
        try (var fixture = new Fixture("F8_E4M3");
             var descending = new PagedTensorReader(4096, 16);
             var ascending = new PagedTensorReader(4096, 16)) {
            var descendingTable = fixture.table(descending);
            var ascendingTable = fixture.table(ascending);
            long beforeDescending = descending.statistics().misses();
            long beforeAscending = ascending.statistics().misses();
            float[] reversed = descendingTable.fetchRows(new long[]{11, 9, 7, 5, 3, 1});
            float[] forward = ascendingTable.fetchRows(new long[]{1, 3, 5, 7, 9, 11});
            assertEquals(ascending.statistics().misses() - beforeAscending,
                    descending.statistics().misses() - beforeDescending);
            // Same rows, reversed request order, reversed result blocks.
            for (int i = 0; i < 6; i++)
                assertBitEqual(Arrays.copyOfRange(forward, i * WIDTH, i * WIDTH + WIDTH),
                        Arrays.copyOfRange(reversed, (5 - i) * WIDTH, (5 - i) * WIDTH + WIDTH));
        }
    }

    @Test public void prefetchWarmsPagesSoTheFetchMissesNothing() throws Exception {
        try (var fixture = new Fixture("F8_E4M3"); var reader = new PagedTensorReader(4096, 16)) {
            var table = fixture.table(reader);
            long[] rows = {11, 5, 5, 0, 6};
            long before = reader.statistics().misses();
            table.prefetchRows(rows);
            long warmed = reader.statistics().misses();
            assertTrue(warmed > before);
            assertBitEqual(perRowReference(table, rows), table.fetchRows(rows));
            assertEquals(warmed, reader.statistics().misses());
            table.prefetchRows(new int[0]);
            assertEquals(warmed, reader.statistics().misses());
        }
    }

    @Test public void dedicatedReaderHasItsOwnBudgetAndEvicts() throws Exception {
        try (var fixture = new Fixture("F8_E4M3"); var metadata = new PagedTensorReader(4096, 16);
             var table = new Qwen4PleTable(fixture.index, metadata, 16L, PREFIX, EOS)) {
            assertEquals(16, table.statistics().budgetBytes());
            long metadataMisses = metadata.statistics().misses();
            table.fetchRows(new long[]{0});
            long first = table.statistics().misses();
            assertTrue(first > 0);
            assertTrue(table.statistics().residentBytes() <= table.statistics().budgetBytes());
            table.fetchRows(new long[]{11});
            long second = table.statistics().misses();
            assertTrue(second > first);
            // The single page was evicted by row 11, so row 0 must be filled again.
            table.fetchRows(new long[]{0});
            assertTrue(table.statistics().misses() > second);
            // Row reads never touched the dense/metadata reader.
            assertEquals(metadataMisses, metadata.statistics().misses());
        }
    }

    @Test public void fetchValidatesBeforeTouchingDestination() throws Exception {
        try (var fixture = new Fixture("F8_E4M3"); var reader = new PagedTensorReader(4096, 16)) {
            var table = fixture.table(reader);
            float[] destination = new float[2 * WIDTH];
            Arrays.fill(destination, 42f);
            assertThrows(IndexOutOfBoundsException.class, () -> table.fetchRows(new long[]{0, ROWS}, destination));
            for (float value : destination) assertEquals(42f, value, 0);
            assertThrows(IllegalArgumentException.class, () -> table.fetchRows(new long[]{0}, new float[1]));
            assertThrows(IndexOutOfBoundsException.class, () -> table.prefetchRows(new int[]{-1}));
            assertEquals(0, table.fetchRows(new long[0]).length);
        }
    }

    @Test public void plePageCachePropertyResolution() {
        String previous = System.getProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY);
        try {
            assertEquals("jitllm.qwen4.plePageCacheBytes", Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY);
            assertEquals(256L << 20, Qwen4PleTable.DEFAULT_PLE_PAGE_CACHE_BYTES);
            System.clearProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY);
            assertEquals(Qwen4PleTable.DEFAULT_PLE_PAGE_CACHE_BYTES, Qwen4PleTable.plePageCacheBytes());
            System.setProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY, "0");
            assertEquals(Qwen4PleTable.SHARED_PLE_PAGE_CACHE_BYTES, Qwen4PleTable.plePageCacheBytes());
            System.setProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY, "1048576");
            assertEquals(1048576L, Qwen4PleTable.plePageCacheBytes());
            System.setProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY, "-1");
            assertThrows(IllegalArgumentException.class, Qwen4PleTable::plePageCacheBytes);
        } finally {
            if (previous == null) System.clearProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY);
            else System.setProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY, previous);
        }
    }
}

package org.beehive.jitllm.integration.cli;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.beehive.jitllm.format.safetensors.PagedTensorReader;
import org.beehive.jitllm.format.safetensors.SafeTensorIndex;
import org.beehive.jitllm.model.vision.FlashNextImagePreprocessor;
import org.beehive.jitllm.model.vision.FlashNextMrope;
import org.beehive.jitllm.model.vision.FlashNextVisionConfig;
import org.beehive.jitllm.model.vision.FlashNextVisionTower;
import org.beehive.jitllm.model.vision.FlashNextVisionWeights;
import org.beehive.jitllm.model.vision.PreprocessedImage;

/**
 * Parity probe for the Flash-Next vision slice. Loads the checkpoint's vision tower
 * on CPU, preprocesses one image, runs the tower, and prints the grid, merged token
 * count, embedding statistics, and a SHA-256 over the little-endian float32
 * embeddings. Binary dumps (little-endian float32, row-major) support element-wise
 * comparison against an external reference.
 *
 * <pre>
 * FlashNextVisionProbe &lt;checkpointDir&gt; &lt;imagePath&gt; [options]
 *   --rgb WxH             imagePath is raw interleaved uint8 RGB (no smart-resize;
 *                         W and H must be divisible by patch*merge) — exact-parity mode
 *   --preprocess-only     stop after preprocessing (no weight load, no tower)
 *   --dump-patches FILE   write pixel_values ([patches][patchDim])
 *   --dump-patch-embed F  write patch-embed output ([patches][hidden])
 *   --dump-embeddings F   write tower output ([tokens][outHidden])
 * </pre>
 */
public final class FlashNextVisionProbe {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlashNextVisionProbe <checkpointDir> <imagePath> [--rgb WxH] "
                    + "[--preprocess-only] [--dump-patches F] [--dump-patch-embed F] [--dump-embeddings F]");
            System.exit(2);
        }
        Path checkpoint = Path.of(args[0]);
        Path imagePath = Path.of(args[1]);
        int[] rgbSize = null;
        boolean preprocessOnly = false;
        Path dumpPatches = null, dumpPatchEmbed = null, dumpEmbeddings = null;
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--rgb" -> rgbSize = parseSize(args[++i]);
                case "--preprocess-only" -> preprocessOnly = true;
                case "--dump-patches" -> dumpPatches = Path.of(args[++i]);
                case "--dump-patch-embed" -> dumpPatchEmbed = Path.of(args[++i]);
                case "--dump-embeddings" -> dumpEmbeddings = Path.of(args[++i]);
                default -> throw new IllegalArgumentException("Unknown option: " + args[i]);
            }
        }

        FlashNextVisionConfig config = FlashNextVisionConfig.load(checkpoint);
        FlashNextImagePreprocessor preprocessor = new FlashNextImagePreprocessor(config);
        byte[] bytes = Files.readAllBytes(imagePath);
        PreprocessedImage image = rgbSize != null
                ? preprocessor.preprocessRgbBytes(bytes, rgbSize[0], rgbSize[1])
                : preprocessor.preprocess(bytes);
        int tokens = FlashNextMrope.tokenCount(image.grid(), config.spatialMergeSize());
        System.out.println("grid=" + image.grid() + " resized=" + image.resizedWidth() + "x" + image.resizedHeight()
                + " patches=" + image.grid().patchCount() + " patchDim=" + image.patchDim() + " tokens=" + tokens);
        printStats("pixel_values", image.pixelValues());
        if (dumpPatches != null) writeFloats(dumpPatches, image.pixelValues());
        if (preprocessOnly) return;

        FlashNextVisionTower tower;
        try (PagedTensorReader reader = new PagedTensorReader(1L << 28, 1 << 20)) {
            SafeTensorIndex index = SafeTensorIndex.open(checkpoint, name -> name.startsWith("model.visual."));
            tower = new FlashNextVisionTower(FlashNextVisionWeights.load(config, index, reader));
        }
        if (dumpPatchEmbed != null) {
            float[][] patchEmbed = tower.patchEmbed(image);
            printStats("patch_embed", patchEmbed);
            writeRows(dumpPatchEmbed, patchEmbed);
        }
        long started = System.nanoTime();
        float[][] embeddings = tower.forward(image);
        double seconds = (System.nanoTime() - started) / 1e9;
        System.out.println("embeddings=" + embeddings.length + "x" + embeddings[0].length
                + " forwardSeconds=" + String.format("%.2f", seconds));
        printStats("embeddings", embeddings);
        if (dumpEmbeddings != null) writeRows(dumpEmbeddings, embeddings);
        System.out.println("embeddingsSha256=" + sha256(embeddings));
    }

    private static int[] parseSize(String value) {
        String[] parts = value.split("x");
        if (parts.length != 2) throw new IllegalArgumentException("Expected WxH, got: " + value);
        return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
    }

    private static void printStats(String name, float[] values) {
        double sum = 0, sumSq = 0;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            sum += value;
            sumSq += value * (double) value;
            if (value < min) min = value;
            if (value > max) max = value;
        }
        System.out.println(name + ": n=" + values.length + " mean=" + (sum / values.length)
                + " rms=" + Math.sqrt(sumSq / values.length) + " min=" + min + " max=" + max
                + " first=" + values[0] + " last=" + values[values.length - 1]);
    }

    private static void printStats(String name, float[][] rows) {
        int total = 0;
        for (float[] row : rows) total += row.length;
        float[] flat = new float[total];
        int index = 0;
        for (float[] row : rows) {
            System.arraycopy(row, 0, flat, index, row.length);
            index += row.length;
        }
        printStats(name + "[" + rows.length + "x" + rows[0].length + "]", flat);
    }

    private static void writeFloats(Path file, float[] values) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) buffer.putFloat(value);
        Files.write(file, buffer.array());
    }

    private static void writeRows(Path file, float[][] rows) throws IOException {
        int total = 0;
        for (float[] row : rows) total += row.length;
        float[] flat = new float[total];
        int index = 0;
        for (float[] row : rows) {
            System.arraycopy(row, 0, flat, index, row.length);
            index += row.length;
        }
        writeFloats(file, flat);
    }

    private static String sha256(float[][] rows) throws NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        ByteBuffer buffer = ByteBuffer.allocate(8192 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] row : rows)
            for (float value : row) {
                if (!buffer.hasRemaining()) {
                    digest.update(buffer.array());
                    buffer.clear();
                }
                buffer.putFloat(value);
            }
        digest.update(buffer.array(), 0, buffer.position());
        return HexFormat.of().formatHex(digest.digest());
    }
}

package org.beehive.jitllm.model.vision;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.format.safetensors.MetadataJson;

/**
 * Vision-tower and image-processor dimensions of the Qwen3.8-Flash-Next
 * ({@code qwen4_exp}) checkpoint, read from {@code config.json} {@code vision_config}
 * plus {@code preprocessor_config.json}.
 *
 * <p>Verified against the staged NVFP4 checkpoint: depth 27, hidden 1152, 16 heads
 * (head dim 72), intermediate 4304, patch 16, temporal patch 2, spatial merge 2,
 * output hidden 2560 (the language-model hidden size), 2304 position embeddings
 * (a 48x48 table), activation {@code gelu_pytorch_tanh}; processor mean/std 0.5,
 * min pixels 65536, max pixels 16777216. This is the Qwen3-VL vision architecture;
 * {@code deepstack_visual_indexes} must be empty for this checkpoint.
 */
public record FlashNextVisionConfig(
        int depth, int hiddenSize, int numHeads, int intermediateSize, int inChannels,
        int patchSize, int temporalPatchSize, int spatialMergeSize, int outHiddenSize,
        int numPositionEmbeddings, String hiddenAct,
        float[] imageMean, float[] imageStd, int minPixels, int maxPixels,
        int imageTokenId, int videoTokenId, int visionStartTokenId, int visionEndTokenId) {

    public FlashNextVisionConfig {
        imageMean = imageMean.clone();
        imageStd = imageStd.clone();
        int[] positive = {depth, hiddenSize, numHeads, intermediateSize, inChannels, patchSize,
                temporalPatchSize, spatialMergeSize, outHiddenSize, numPositionEmbeddings, minPixels, maxPixels};
        for (int value : positive) if (value <= 0) throw new IllegalArgumentException("Vision dimensions must be positive");
        if (hiddenSize % numHeads != 0) throw new IllegalArgumentException("Vision hidden size not divisible by heads");
        if ((hiddenSize / numHeads) % 2 != 0) throw new IllegalArgumentException("Vision head dim must be even for rotary embedding");
        int side = (int) Math.round(Math.sqrt(numPositionEmbeddings));
        if (side * side != numPositionEmbeddings)
            throw new IllegalArgumentException("num_position_embeddings must be a perfect square: " + numPositionEmbeddings);
        if (imageMean.length != inChannels || imageStd.length != inChannels)
            throw new IllegalArgumentException("image_mean/image_std must have one entry per channel");
        for (float std : imageStd) if (!(std > 0) || !Float.isFinite(std)) throw new IllegalArgumentException("Invalid image_std");
        if (!"gelu_pytorch_tanh".equals(hiddenAct))
            throw new IllegalArgumentException("Unsupported vision activation: " + hiddenAct);
        if (minPixels > maxPixels) throw new IllegalArgumentException("min_pixels exceeds max_pixels");
    }

    public int headDim() { return hiddenSize / numHeads; }

    /** Rotary dimension of the vision attention: half the head dim (angles repeat over both halves). */
    public int rotaryDim() { return headDim() / 2; }

    /** Side length of the square learned position-embedding table (48 for this checkpoint). */
    public int numGridPerSide() { return (int) Math.round(Math.sqrt(numPositionEmbeddings)); }

    /** Smart-resize alignment factor: patch_size * merge_size (32 for this checkpoint). */
    public int resizeFactor() { return Math.multiplyExact(patchSize, spatialMergeSize); }

    /** Flattened patch width: in_channels * temporal_patch * patch * patch (1536 here). */
    public int patchDim() {
        return Math.multiplyExact(inChannels, Math.multiplyExact(temporalPatchSize, Math.multiplyExact(patchSize, patchSize)));
    }

    /** Width after grouping one spatial-merge block: hidden * merge^2 (4608 here). */
    public int mergedDim() { return Math.multiplyExact(hiddenSize, spatialMergeSize * spatialMergeSize); }

    /** Number of language-model tokens one image with this grid expands to. */
    public int tokenCount(ImageGrid grid) { return grid.tokenCount(spatialMergeSize); }

    public static FlashNextVisionConfig load(Path directory) throws IOException {
        Map<String, Object> root = MetadataJson.parseObject(Files.readString(directory.resolve("config.json")));
        if (!"qwen4_exp".equals(root.get("model_type"))) throw new IOException("Expected qwen4_exp checkpoint");
        Map<String, Object> vision = object(root.get("vision_config"), "vision_config");
        if (vision.get("deepstack_visual_indexes") instanceof List<?> deep && !deep.isEmpty())
            throw new IOException("deepstack_visual_indexes is not empty; deepstack mergers are not implemented");
        Map<String, Object> pre = preprocessor(directory);
        Map<String, Object> size = object(pre.get("size"), "preprocessor size");
        return new FlashNextVisionConfig(
                integer(vision, "depth"), integer(vision, "hidden_size"), integer(vision, "num_heads"),
                integer(vision, "intermediate_size"), integer(vision, "in_channels"),
                integer(vision, "patch_size"), integer(vision, "temporal_patch_size"),
                integer(vision, "spatial_merge_size"), integer(vision, "out_hidden_size"),
                integer(vision, "num_position_embeddings"), (String) vision.get("hidden_act"),
                floats(pre.get("image_mean"), "image_mean"), floats(pre.get("image_std"), "image_std"),
                integer(size, "shortest_edge"), integer(size, "longest_edge"),
                integer(root, "image_token_id"), integer(root, "video_token_id"),
                integer(root, "vision_start_token_id"), integer(root, "vision_end_token_id"));
    }

    private static Map<String, Object> preprocessor(Path directory) throws IOException {
        Path flat = directory.resolve("preprocessor_config.json");
        if (Files.exists(flat)) {
            Map<String, Object> parsed = MetadataJson.parseObject(Files.readString(flat));
            if (parsed.containsKey("size")) return parsed;
        }
        Map<String, Object> processor = MetadataJson.parseObject(Files.readString(directory.resolve("processor_config.json")));
        return object(processor.get("image_processor"), "image_processor");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> object(Object value, String name) throws IOException {
        if (!(value instanceof Map)) throw new IOException("Missing config object: " + name);
        return (Map<String, Object>) value;
    }

    private static float[] floats(Object value, String name) throws IOException {
        if (!(value instanceof List<?> list) || list.isEmpty()) throw new IOException("Missing config list: " + name);
        float[] result = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Number n)) throw new IOException("Invalid number in " + name);
            result[i] = n.floatValue();
        }
        return result;
    }

    private static int integer(Map<String, Object> map, String key) throws IOException {
        if (!(map.get(key) instanceof Number n) || n.doubleValue() != Math.rint(n.doubleValue())
                || n.doubleValue() < Integer.MIN_VALUE || n.doubleValue() > Integer.MAX_VALUE)
            throw new IOException("Missing or invalid integer config: " + key);
        return n.intValue();
    }
}

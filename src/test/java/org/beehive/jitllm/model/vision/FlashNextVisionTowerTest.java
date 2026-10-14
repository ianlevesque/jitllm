package org.beehive.jitllm.model.vision;

import static org.junit.Assert.*;
import java.util.List;
import org.junit.Test;

/** Structural tower checks on a tiny synthetic configuration (real-checkpoint parity lives in FlashNextVisionProbe runs). */
public final class FlashNextVisionTowerTest {

    private static short bf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    private static short[] bf16s(float... values) {
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) result[i] = bf16(values[i]);
        return result;
    }

    /** All-zero weights except merger fc2 bias: the tower must emit that bias on every token. */
    @Test public void zeroWeightsPropagateMergerBiasExactly() {
        FlashNextVisionConfig config = FlashNextImagePreprocessorTest.tinyConfig();
        int hidden = 8, inter = 12, merged = 32, out = 4, patchDim = 24;
        float[] patchBias = new float[hidden];
        for (int i = 0; i < hidden; i++) patchBias[i] = i + 1;
        var block = new FlashNextVisionWeights.Block(
                new short[3 * hidden * hidden], new float[3 * hidden],
                new short[hidden * hidden], new float[hidden],
                new short[inter * hidden], new float[inter],
                new short[hidden * inter], new float[hidden],
                filled(hidden, 1f), new float[hidden], filled(hidden, 1f), new float[hidden]);
        float[] fc2Bias = {1f, -2f, 3.5f, 0f};
        var weights = new FlashNextVisionWeights(config,
                new short[hidden * patchDim], patchBias, new float[16 * hidden], List.of(block),
                filled(hidden, 1f), new float[hidden],
                new short[merged * merged], new float[merged],
                new short[out * merged], fc2Bias);
        FlashNextVisionTower tower = new FlashNextVisionTower(weights);
        PreprocessedImage image = new FlashNextImagePreprocessor(config)
                .preprocessPixels(new int[4 * 4], 4, 4);
        assertEquals(1, tower.tokenCount(image.grid()));
        float[][] patchEmbed = tower.patchEmbed(image);
        assertEquals(4, patchEmbed.length);
        assertArrayEquals(patchBias, patchEmbed[0], 0f);
        float[][] embeddings = tower.forward(image);
        assertEquals(1, embeddings.length);
        assertArrayEquals(fc2Bias, embeddings[0], 0f);
    }

    /** One-hot patch-embed rows verify the flattened (channel, temporal, row, col) layout end to end. */
    @Test public void patchEmbedReadsFlattenedLayout() {
        FlashNextVisionConfig config = FlashNextImagePreprocessorTest.tinyConfig();
        int hidden = 8, inter = 12, merged = 32, out = 4, patchDim = 24;
        short[] patchWeight = new short[hidden * patchDim];
        // Row 0 <- (c=0,t=1,ph=1,pw=0) index ((0*2+1)*2+1)*2+0 = 6; row 1 <- (c=2,t=0,ph=0,pw=1) = 17.
        patchWeight[0 * patchDim + 6] = bf16(1f);
        patchWeight[1 * patchDim + 17] = bf16(1f);
        var block = new FlashNextVisionWeights.Block(
                new short[3 * hidden * hidden], new float[3 * hidden],
                new short[hidden * hidden], new float[hidden],
                new short[inter * hidden], new float[inter],
                new short[hidden * inter], new float[hidden],
                filled(hidden, 1f), new float[hidden], filled(hidden, 1f), new float[hidden]);
        var weights = new FlashNextVisionWeights(config, patchWeight, new float[hidden],
                new float[16 * hidden], List.of(block),
                filled(hidden, 1f), new float[hidden],
                new short[merged * merged], new float[merged],
                new short[out * merged], new float[out]);
        int[] argb = new int[16];
        for (int i = 0; i < 16; i++) argb[i] = 0xFF000000 | (i * 16) << 16 | 0x40;
        // Pixel (y=1,x=0) is index 4 -> R = 64; B plane constant 0x40 at (y=0,x=1).
        PreprocessedImage image = new FlashNextImagePreprocessor(config).preprocessPixels(argb, 4, 4);
        float[][] patchEmbed = new FlashNextVisionTower(weights).patchEmbed(image);
        assertEquals((64 / 255.0f - 0.5f) / 0.5f, patchEmbed[0][0], 0f);
        assertEquals((0x40 / 255.0f - 0.5f) / 0.5f, patchEmbed[0][1], 0f);
    }

    @Test public void activationsMatchReferenceDefinitions() {
        assertEquals(0f, FlashNextVisionTower.geluTanh(0f), 0f);
        assertEquals(0f, FlashNextVisionTower.geluErf(0f), 0f);
        // torch.nn.functional.gelu(1.0) = 0.8413447; tanh approximation = 0.8411920.
        assertEquals(0.8413447f, FlashNextVisionTower.geluErf(1f), 1e-6f);
        assertEquals(0.8411920f, FlashNextVisionTower.geluTanh(1f), 1e-6f);
        assertEquals(-0.1586553f, FlashNextVisionTower.geluErf(-1f), 1e-6f);
    }

    @Test public void weightShapeValidationRejectsTruncatedBlock() {
        FlashNextVisionConfig config = FlashNextImagePreprocessorTest.tinyConfig();
        var block = new FlashNextVisionWeights.Block(
                new short[3], new float[24], new short[64], new float[8],
                new short[96], new float[12], new short[96], new float[8],
                new float[8], new float[8], new float[8], new float[8]);
        assertThrows(IllegalArgumentException.class, () -> new FlashNextVisionWeights(config,
                new short[192], new float[8], new float[128], List.of(block),
                new float[8], new float[8], new short[1024], new float[32], new short[128], new float[4]));
    }

    private static float[] filled(int length, float value) {
        float[] result = new float[length];
        java.util.Arrays.fill(result, value);
        return result;
    }
}

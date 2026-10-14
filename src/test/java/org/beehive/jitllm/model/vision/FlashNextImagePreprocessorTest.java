package org.beehive.jitllm.model.vision;

import static org.junit.Assert.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import javax.imageio.ImageIO;
import org.junit.Test;

public final class FlashNextImagePreprocessorTest {

    /** Tiny config: patch 2, temporal 2, merge 2 (factor 4), patchDim 24. */
    static FlashNextVisionConfig tinyConfig() {
        return new FlashNextVisionConfig(1, 8, 2, 12, 3, 2, 2, 2, 4, 16,
                "gelu_pytorch_tanh", new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f},
                16, 1 << 20, 10, 11, 12, 13);
    }

    private static float normalized(int channel) {
        return (channel / 255.0f - 0.5f) / 0.5f;
    }

    @Test public void smartResizeMatchesReferenceTable() {
        // Expected values computed with transformers 5.12 smart_resize(factor=32,
        // min_pixels=65536, max_pixels=16777216) — the checkpoint's processor settings.
        int[][] cases = {
                {256, 256, 256, 256}, {100, 100, 256, 256}, {64, 96, 224, 320},
                {480, 640, 480, 640}, {17, 23, 224, 320}, {4096, 4096, 4096, 4096},
                {5000, 3000, 4992, 3008}, {32, 32, 256, 256}, {1000, 37, 1344, 64},
                {224, 224, 256, 256}};
        for (int[] c : cases) {
            int[] got = FlashNextImagePreprocessor.smartResize(c[0], c[1], 32, 65536, 16777216);
            assertArrayEquals("smartResize(" + c[0] + "," + c[1] + ")", new int[]{c[2], c[3]}, got);
        }
        assertThrows(IllegalArgumentException.class,
                () -> FlashNextImagePreprocessor.smartResize(1, 1000, 32, 65536, 16777216));
    }

    @Test public void patchifyUsesMergeBlockMajorOrderAndChannelTemporalLayout() {
        FlashNextImagePreprocessor processor = new FlashNextImagePreprocessor(tinyConfig());
        int width = 8, height = 4; // grid (1,2,4), two merge blocks side by side
        int[] argb = new int[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int value = y * width + x;
                argb[y * width + x] = 0xFF000000 | value << 16 | (255 - value) << 8 | 128;
            }
        PreprocessedImage image = processor.preprocessPixels(argb, width, height);
        assertEquals(new ImageGrid(1, 2, 4), image.grid());
        assertEquals(24, image.patchDim());
        assertEquals(2, FlashNextMrope.tokenCount(image.grid(), 2));
        float[] pv = image.pixelValues();
        // Patch 0 is image patch (row 0, col 0): R channel, t=0, patch rows/cols.
        assertEquals(normalized(0), pv[0], 0f);
        assertEquals(normalized(1), pv[1], 0f);
        assertEquals(normalized(8), pv[2], 0f);
        // Temporal copy t=1 duplicates t=0 (offset one patch plane: 2*2).
        assertEquals(normalized(0), pv[4], 0f);
        assertEquals(normalized(9), pv[7], 0f);
        // G channel follows both temporal copies of R (offset 2 planes = 8).
        assertEquals(normalized(255), pv[8], 0f);
        // Decisive ordering check: patch 2 must be (row 1, col 0) — inside the first
        // merge block — not raster-order (row 0, col 2).
        assertEquals(normalized(16), pv[2 * 24], 0f);
        // Patch 4 opens the second merge block: (row 0, col 2).
        assertEquals(normalized(4), pv[4 * 24], 0f);
    }

    @Test public void patchifyRejectsUnalignedSize() {
        FlashNextImagePreprocessor processor = new FlashNextImagePreprocessor(tinyConfig());
        assertThrows(IllegalArgumentException.class,
                () -> processor.preprocessPixels(new int[6 * 4], 6, 4));
    }

    @Test public void encodedImageDecodesSmartResizesAndPatchifies() throws Exception {
        FlashNextVisionConfig config = new FlashNextVisionConfig(1, 8, 2, 12, 3, 16, 2, 2, 4, 2304,
                "gelu_pytorch_tanh", new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f},
                65536, 16777216, 10, 11, 12, 13);
        FlashNextImagePreprocessor processor = new FlashNextImagePreprocessor(config);
        BufferedImage source = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 100; y++)
            for (int x = 0; x < 100; x++) source.setRGB(x, y, (x * 2 << 16) | (y * 2 << 8) | 64);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        assertTrue(ImageIO.write(source, "png", bytes));
        PreprocessedImage image = processor.preprocess(bytes.toByteArray());
        assertEquals(256, image.resizedWidth());
        assertEquals(256, image.resizedHeight());
        assertEquals(new ImageGrid(1, 16, 16), image.grid());
        assertEquals(64, FlashNextMrope.tokenCount(image.grid(), 2));
        assertEquals(1536, image.patchDim());
        for (float value : image.pixelValues()) assertTrue(value >= -1.0001f && value <= 1.0001f);
    }
}

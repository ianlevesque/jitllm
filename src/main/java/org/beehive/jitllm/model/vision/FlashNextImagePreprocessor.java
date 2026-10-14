package org.beehive.jitllm.model.vision;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Objects;
import javax.imageio.ImageIO;

/**
 * CPU image preprocessing matching Hugging Face {@code Qwen2VLImageProcessor}
 * (the {@code Qwen2VLImageProcessorFast} used by the checkpoint's
 * {@code preprocessor_config.json}) and SGLang's {@code qwen_vl} processor:
 * smart-resize by factor {@code patch_size * merge_size} into
 * {@code [min_pixels, max_pixels]}, bicubic resampling, rescale by 1/255,
 * normalize by the configured mean/std (0.5/0.5), then patchify.
 *
 * <p>Patchification reproduces the reference tensor manipulation exactly:
 * reshape to {@code [C, H_m, m, P, W_m, m, P]}, permute to
 * {@code [H_m, W_m, m, m, C, P, P]}, duplicate the frame {@code temporal_patch_size}
 * times on a new axis just before the patch rows, and flatten to
 * {@code [H*W patches][C*T*P*P]}. Grid is {@code (1, H/P, W/P)} for a still image.
 *
 * <p>Known deviation: resampling uses {@link Graphics2D} bicubic interpolation, which
 * is not bit-identical to PIL/torchvision bicubic. When the smart-resized size equals
 * the input size no resampling happens and the output is exact. Alpha is dropped
 * (not composited), matching PIL {@code convert("RGB")} channel semantics for the
 * no-resize path.
 */
public final class FlashNextImagePreprocessor {

    private final FlashNextVisionConfig config;

    public FlashNextImagePreprocessor(FlashNextVisionConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        if (config.inChannels() != 3) throw new IllegalArgumentException("Only 3-channel RGB input is supported");
    }

    public FlashNextVisionConfig config() { return config; }

    /** Decode encoded image bytes (PNG/JPEG/...) and run the full reference pipeline. */
    public PreprocessedImage preprocess(byte[] encodedImage) throws IOException {
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(encodedImage));
        if (image == null) throw new IOException("Undecodable image bytes (" + encodedImage.length + " bytes)");
        int[] resized = smartResize(image.getHeight(), image.getWidth(),
                config.resizeFactor(), config.minPixels(), config.maxPixels());
        BufferedImage rgb = toRgb(image, resized[1], resized[0]);
        int width = rgb.getWidth(), height = rgb.getHeight();
        int[] argb = rgb.getRGB(0, 0, width, height, null, 0, width);
        return patchify(argb, width, height);
    }

    /**
     * Patchify already-sized pixels without smart-resize. Height and width must be
     * divisible by {@code patch_size * merge_size}. Intended for parity probes and
     * callers that resized with their own (reference-exact) resampler.
     * Pixels are packed {@code 0xAARRGGBB} row-major; alpha is ignored.
     */
    public PreprocessedImage preprocessPixels(int[] argbRowMajor, int width, int height) {
        Objects.requireNonNull(argbRowMajor, "argbRowMajor");
        if (width <= 0 || height <= 0 || argbRowMajor.length != Math.multiplyExact(width, height))
            throw new IllegalArgumentException("Pixel array does not match " + width + "x" + height);
        int factor = config.resizeFactor();
        if (height % factor != 0 || width % factor != 0)
            throw new IllegalArgumentException("Image " + width + "x" + height + " is not divisible by factor " + factor);
        return patchify(argbRowMajor.clone(), width, height);
    }

    /** Convenience form taking interleaved unsigned RGB bytes, row-major. */
    public PreprocessedImage preprocessRgbBytes(byte[] rgbRowMajor, int width, int height) {
        Objects.requireNonNull(rgbRowMajor, "rgbRowMajor");
        if (rgbRowMajor.length != Math.multiplyExact(width, Math.multiplyExact(height, 3)))
            throw new IllegalArgumentException("RGB byte array does not match " + width + "x" + height);
        int[] argb = new int[width * height];
        for (int i = 0; i < argb.length; i++)
            argb[i] = 0xFF000000 | (rgbRowMajor[3 * i] & 0xFF) << 16
                    | (rgbRowMajor[3 * i + 1] & 0xFF) << 8 | (rgbRowMajor[3 * i + 2] & 0xFF);
        return preprocessPixels(argb, width, height);
    }

    /**
     * Reference {@code smart_resize}: both dimensions divisible by {@code factor},
     * total pixels within {@code [minPixels, maxPixels]}, aspect ratio preserved as
     * closely as possible. Returns {@code {height, width}}. Uses round-half-even to
     * match Python {@code round()}.
     */
    public static int[] smartResize(int height, int width, int factor, int minPixels, int maxPixels) {
        if (height <= 0 || width <= 0 || factor <= 0) throw new IllegalArgumentException("Invalid smart-resize input");
        if ((double) Math.max(height, width) / Math.min(height, width) > 200.0)
            throw new IllegalArgumentException("Absolute aspect ratio must be smaller than 200, got "
                    + ((double) Math.max(height, width) / Math.min(height, width)));
        long hBar = Math.round(Math.rint((double) height / factor)) * factor;
        long wBar = Math.round(Math.rint((double) width / factor)) * factor;
        if (hBar * wBar > maxPixels) {
            double beta = Math.sqrt((double) height * width / maxPixels);
            hBar = Math.max(factor, (long) Math.floor(height / beta / factor) * factor);
            wBar = Math.max(factor, (long) Math.floor(width / beta / factor) * factor);
        } else if (hBar * wBar < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) height * width));
            hBar = (long) Math.ceil(height * beta / factor) * factor;
            wBar = (long) Math.ceil(width * beta / factor) * factor;
        }
        return new int[]{Math.toIntExact(hBar), Math.toIntExact(wBar)};
    }

    private PreprocessedImage patchify(int[] argb, int width, int height) {
        int patch = config.patchSize(), merge = config.spatialMergeSize(), temporal = config.temporalPatchSize();
        int gridH = height / patch, gridW = width / patch;
        int patchDim = config.patchDim();
        float[] pixels = new float[Math.multiplyExact(gridH * gridW, patchDim)];
        float[] mean = config.imageMean(), std = config.imageStd();
        int token = 0;
        for (int blockRow = 0; blockRow < gridH / merge; blockRow++)
            for (int blockCol = 0; blockCol < gridW / merge; blockCol++)
                for (int inRow = 0; inRow < merge; inRow++)
                    for (int inCol = 0; inCol < merge; inCol++) {
                        int patchRow = blockRow * merge + inRow, patchCol = blockCol * merge + inCol;
                        int base = token++ * patchDim;
                        for (int c = 0; c < 3; c++)
                            for (int t = 0; t < temporal; t++)
                                for (int ph = 0; ph < patch; ph++) {
                                    int y = patchRow * patch + ph;
                                    int out = base + ((c * temporal + t) * patch + ph) * patch;
                                    for (int pw = 0; pw < patch; pw++) {
                                        int x = patchCol * patch + pw;
                                        int rgb = argb[y * width + x];
                                        int channel = c == 0 ? (rgb >>> 16) & 0xFF : c == 1 ? (rgb >>> 8) & 0xFF : rgb & 0xFF;
                                        // Reference order: rescale (1/255) then (x - mean) / std.
                                        pixels[out + pw] = (channel / 255.0f - mean[c]) / std[c];
                                    }
                                }
                    }
        return new PreprocessedImage(pixels, new ImageGrid(1, gridH, gridW), height, width);
    }

    private static BufferedImage toRgb(BufferedImage source, int targetWidth, int targetHeight) {
        BufferedImage rgb = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        int[] raw = source.getRGB(0, 0, source.getWidth(), source.getHeight(), null, 0, source.getWidth());
        rgb.setRGB(0, 0, source.getWidth(), source.getHeight(), raw, 0, source.getWidth());
        if (targetWidth == rgb.getWidth() && targetHeight == rgb.getHeight()) return rgb;
        BufferedImage scaled = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = scaled.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            g.drawImage(rgb, 0, 0, targetWidth, targetHeight, null);
        } finally {
            g.dispose();
        }
        return scaled;
    }
}

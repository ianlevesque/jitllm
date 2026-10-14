package org.beehive.jitllm.model.vision;

/**
 * Patch grid of one image (or one temporal group of a video) in the layout used by
 * Hugging Face {@code image_grid_thw}: {@code t} temporal patch groups, {@code h} and
 * {@code w} patch rows/columns <em>before</em> spatial merging. A still image always
 * has {@code t == 1}; its single frame is duplicated inside each flattened patch to
 * fill the temporal patch size (see {@link FlashNextImagePreprocessor}).
 */
public record ImageGrid(int t, int h, int w) {

    public ImageGrid {
        if (t <= 0 || h <= 0 || w <= 0)
            throw new IllegalArgumentException("Image grid dimensions must be positive: " + t + "," + h + "," + w);
    }

    /** Number of flattened patches fed to the vision tower. */
    public int patchCount() {
        return Math.multiplyExact(t, Math.multiplyExact(h, w));
    }

    /** Number of language-model tokens this image occupies after spatial merging. */
    public int tokenCount(int spatialMergeSize) {
        if (spatialMergeSize <= 0 || h % spatialMergeSize != 0 || w % spatialMergeSize != 0)
            throw new IllegalArgumentException("Grid " + this + " is not divisible by merge size " + spatialMergeSize);
        return Math.multiplyExact(t, Math.multiplyExact(h / spatialMergeSize, w / spatialMergeSize));
    }

    @Override public String toString() {
        return "(" + t + "," + h + "," + w + ")";
    }
}

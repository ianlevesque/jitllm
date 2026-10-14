package org.beehive.jitllm.model.vision;

import java.util.Objects;

/**
 * Output of {@link FlashNextImagePreprocessor}: flattened, normalized patches in the
 * exact Hugging Face {@code Qwen2VLImageProcessor} {@code pixel_values} layout.
 *
 * <p>{@code pixelValues} is row-major with shape {@code [grid.patchCount()][patchDim]}
 * where {@code patchDim = inChannels * temporalPatchSize * patchSize * patchSize}.
 * Patch order is spatial-merge-block-major (merged block row, merged block column,
 * row inside block, column inside block); inside one patch the order is channel,
 * temporal copy, patch row, patch column. The vision tower consumes this layout
 * directly; do not reorder it.
 */
public record PreprocessedImage(float[] pixelValues, ImageGrid grid, int resizedHeight, int resizedWidth) {

    public PreprocessedImage {
        Objects.requireNonNull(pixelValues, "pixelValues");
        Objects.requireNonNull(grid, "grid");
        if (resizedHeight <= 0 || resizedWidth <= 0)
            throw new IllegalArgumentException("Resized dimensions must be positive");
        if (pixelValues.length == 0 || pixelValues.length % grid.patchCount() != 0)
            throw new IllegalArgumentException("pixelValues length " + pixelValues.length
                    + " does not divide by patch count " + grid.patchCount());
    }

    /** Flattened width of one patch (channels * temporal * patch * patch). */
    public int patchDim() {
        return pixelValues.length / grid.patchCount();
    }
}

package org.beehive.jitllm.model.vision;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Objects;
import java.util.Queue;

/**
 * Language-model MRoPE position computation for mixed text/image sequences,
 * reproducing Hugging Face Qwen3-VL {@code get_rope_index} /
 * {@code get_vision_position_ids} (the scheme the {@code qwen4_exp} trunk was
 * trained with; its {@code rope_parameters} are {@code mrope_section [11,11,10]},
 * {@code mrope_interleaved true}).
 *
 * <p>Text tokens receive {@code (p, p, p)} with {@code p} advancing by one per token.
 * The merged image tokens of one image receive, in row-major merged-grid order,
 * {@code (start, start + row, start + col)} for a still image; afterwards the running
 * position advances by {@code max(gridH, gridW) / merge_size} (not by the token
 * count). {@code positionDelta} is {@code maxPosition + 1 - sequenceLength}, the
 * offset the reference adds to positions of tokens generated after the prompt.
 */
public final class FlashNextMrope {

    private FlashNextMrope() {}

    /** MRoPE positions: {@code positions[axis][token]} with axes (t, h, w), plus the decode delta. */
    public record RopeIndex(int[][] positions, int positionDelta) {}

    /** One image expansion inside a token sequence: start offset and its grid. */
    public record ImageSegment(int startToken, ImageGrid grid) {}

    /** Number of language-model tokens one image occupies after spatial merging. */
    public static int tokenCount(ImageGrid grid, int spatialMergeSize) {
        return grid.tokenCount(spatialMergeSize);
    }

    /**
     * Positions of one image's merged tokens, {@code [3][tokenCount]} axes (t, h, w),
     * following the reference repeat pattern (temporal frames outermost, then merged
     * rows, merged columns fastest).
     */
    public static int[][] visionPositions(int startPosition, ImageGrid grid, int spatialMergeSize) {
        Objects.requireNonNull(grid, "grid");
        int llmT = grid.t(), llmH = grid.h() / spatialMergeSize, llmW = grid.w() / spatialMergeSize;
        if (grid.h() % spatialMergeSize != 0 || grid.w() % spatialMergeSize != 0)
            throw new IllegalArgumentException("Grid " + grid + " not divisible by merge size " + spatialMergeSize);
        int[][] positions = new int[3][llmT * llmH * llmW];
        int index = 0;
        for (int t = 0; t < llmT; t++)
            for (int row = 0; row < llmH; row++)
                for (int col = 0; col < llmW; col++) {
                    positions[0][index] = startPosition + t;
                    positions[1][index] = startPosition + row;
                    positions[2][index] = startPosition + col;
                    index++;
                }
        return positions;
    }

    /**
     * Full-sequence MRoPE positions. {@code tokenTypes} uses the reference encoding
     * (0 = text, 1 = image, 2 = video); consecutive image/video runs consume grids
     * from the queues in order and each run's length must equal that grid's merged
     * token count. Video grids are first split into per-frame {@code t == 1} grids,
     * as in the reference.
     */
    public static RopeIndex ropeIndex(int[] tokenTypes, List<ImageGrid> imageGrids,
                                      List<ImageGrid> videoGrids, int spatialMergeSize) {
        Objects.requireNonNull(tokenTypes, "tokenTypes");
        Queue<ImageGrid> images = new ArrayDeque<>(imageGrids == null ? List.of() : imageGrids);
        Queue<ImageGrid> videos = new ArrayDeque<>();
        if (videoGrids != null)
            for (ImageGrid grid : videoGrids)
                for (int frame = 0; frame < grid.t(); frame++) videos.add(new ImageGrid(1, grid.h(), grid.w()));
        int[][] positions = new int[3][tokenTypes.length];
        int currentPos = 0, i = 0, maxPosition = -1;
        while (i < tokenTypes.length) {
            int type = tokenTypes[i];
            int start = i;
            while (i < tokenTypes.length && tokenTypes[i] == type) i++;
            int length = i - start;
            if (type == 0) {
                for (int j = 0; j < length; j++)
                    for (int axis = 0; axis < 3; axis++) positions[axis][start + j] = currentPos + j;
                if (length > 0) maxPosition = Math.max(maxPosition, currentPos + length - 1);
                currentPos += length;
            } else if (type == 1 || type == 2) {
                ImageGrid grid = (type == 1 ? images : videos).poll();
                if (grid == null) throw new IllegalArgumentException("No grid left for token run at " + start);
                if (grid.tokenCount(spatialMergeSize) != length)
                    throw new IllegalArgumentException("Token run at " + start + " has length " + length
                            + " but grid " + grid + " expands to " + grid.tokenCount(spatialMergeSize));
                int[][] vision = visionPositions(currentPos, grid, spatialMergeSize);
                for (int axis = 0; axis < 3; axis++)
                    System.arraycopy(vision[axis], 0, positions[axis], start, length);
                for (int value : vision[0]) maxPosition = Math.max(maxPosition, value);
                for (int value : vision[1]) maxPosition = Math.max(maxPosition, value);
                for (int value : vision[2]) maxPosition = Math.max(maxPosition, value);
                currentPos += Math.max(grid.h(), grid.w()) / spatialMergeSize;
            } else {
                throw new IllegalArgumentException("Unknown token type " + type + " at " + start);
            }
        }
        if (!images.isEmpty() || !videos.isEmpty())
            throw new IllegalArgumentException("Unused image/video grids remain");
        return new RopeIndex(positions, maxPosition + 1 - tokenTypes.length);
    }

    /**
     * Convenience form for a token-id sequence already expanded with
     * {@code imageTokenId} placeholders: derives token types and validates that each
     * placeholder run matches the grids in order.
     */
    public static RopeIndex ropeIndexForTokens(int[] tokenIds, int imageTokenId,
                                               List<ImageGrid> imageGrids, int spatialMergeSize) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        int[] types = new int[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) types[i] = tokenIds[i] == imageTokenId ? 1 : 0;
        return ropeIndex(types, imageGrids, List.of(), spatialMergeSize);
    }
}

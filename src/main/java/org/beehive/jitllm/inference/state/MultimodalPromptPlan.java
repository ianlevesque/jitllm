package org.beehive.jitllm.inference.state;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.model.vision.FlashNextMrope;
import org.beehive.jitllm.model.vision.ImageGrid;

/**
 * Everything one image-bearing prompt needs beyond its token ids, indexed by absolute
 * sequence position: the tower row that replaces the token embedding at each image
 * position, and the MRoPE {@code (t, h, w)} triple every position attends with.
 *
 * <p>Immutable once built, which is what makes engine snapshot/rollback with images
 * straightforward: a forward reads the plan at its own position and mutates nothing in
 * it, so restoring the position restores exactly what the next forward will see. Text
 * positions inside the prompt carry their MRoPE triple and no embedding row. Positions
 * past the prompt (decode) are the scalar {@code position + positionDelta} on every
 * axis, where the delta is the reference's {@code maxPosition + 1 - promptLength}: an
 * image advances the running position by {@code max(gridH, gridW) / merge}, not by its
 * token count, so decode continues from where the prompt's positions actually ended.
 */
public final class MultimodalPromptPlan {

    /** One image's span inside the prompt: first image-token position, grid and tower rows. */
    public record ImageSpan(int start, ImageGrid grid, float[][] embeddings) {
        public ImageSpan {
            Objects.requireNonNull(grid, "grid");
            Objects.requireNonNull(embeddings, "embeddings");
            if (start < 0) throw new IllegalArgumentException("Negative image span start: " + start);
        }
    }

    private final int promptLength;
    private final int[][] ropePositions; // [3][promptLength], axes (t, h, w)
    private final float[][] embeddingsByPosition; // [promptLength][], null outside image spans
    private final int positionDelta;
    private final List<ImageSpan> spans;

    private MultimodalPromptPlan(
            int promptLength, int[][] ropePositions, float[][] embeddingsByPosition,
            int positionDelta, List<ImageSpan> spans) {
        this.promptLength = promptLength;
        this.ropePositions = ropePositions;
        this.embeddingsByPosition = embeddingsByPosition;
        this.positionDelta = positionDelta;
        this.spans = spans;
    }

    /**
     * Builds the plan for a fully expanded prompt. {@code spans} must be in prompt order,
     * disjoint, and each span's embedding rows must number exactly its grid's merged token
     * count; the MRoPE computation additionally validates that every {@code imageTokenId}
     * run in {@code tokenIds} matches the grids in order, so a span that disagrees with the
     * tokens it claims to cover is rejected here rather than spliced wrongly.
     */
    public static MultimodalPromptPlan of(
            int[] tokenIds, List<ImageSpan> spans, int imageTokenId, int spatialMergeSize) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(spans, "spans");
        List<ImageGrid> grids = spans.stream().map(ImageSpan::grid).toList();
        FlashNextMrope.RopeIndex rope =
                FlashNextMrope.ropeIndexForTokens(tokenIds, imageTokenId, grids, spatialMergeSize);
        float[][] byPosition = new float[tokenIds.length][];
        int previousEnd = 0;
        for (ImageSpan span : spans) {
            int count = span.grid().tokenCount(spatialMergeSize);
            if (span.embeddings().length != count)
                throw new IllegalArgumentException(
                        "Image at " + span.start() + " has " + span.embeddings().length
                                + " embedding rows but grid " + span.grid() + " expands to " + count);
            if (span.start() < previousEnd || span.start() + count > tokenIds.length)
                throw new IllegalArgumentException("Image spans overlap or leave the prompt at " + span.start());
            for (int i = 0; i < count; i++) {
                if (tokenIds[span.start() + i] != imageTokenId)
                    throw new IllegalArgumentException(
                            "Image span at " + span.start() + " does not cover image tokens");
                float[] row = Objects.requireNonNull(span.embeddings()[i], "embedding row");
                byPosition[span.start() + i] = row.clone();
            }
            previousEnd = span.start() + count;
        }
        return new MultimodalPromptPlan(
                tokenIds.length, rope.positions(), byPosition, rope.positionDelta(), List.copyOf(spans));
    }

    /** Number of prompt tokens this plan covers, image tokens included. */
    public int promptLength() { return promptLength; }

    /** The decode offset: prompt positions end {@code positionDelta} from the token count. */
    public int positionDelta() { return positionDelta; }

    /** The image spans, in prompt order. */
    public List<ImageSpan> spans() { return spans; }

    /**
     * MRoPE triple for one absolute sequence position: the prompt table inside the prompt,
     * the delta-shifted scalar at and past its end. A fresh array per call.
     */
    public int[] ropeAt(int position) {
        if (position < 0) throw new IllegalArgumentException("Negative position: " + position);
        if (position < promptLength)
            return new int[]{
                ropePositions[0][position], ropePositions[1][position], ropePositions[2][position]};
        int decode = position + positionDelta;
        return new int[]{decode, decode, decode};
    }

    /**
     * The tower row for one absolute position, or {@code null} when the position is an
     * ordinary token whose embedding lookup applies. A fresh copy per call.
     */
    public float[] embeddingAt(int position) {
        if (position < 0 || position >= promptLength) return null;
        float[] row = embeddingsByPosition[position];
        return row == null ? null : row.clone();
    }
}

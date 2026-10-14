package org.beehive.jitllm.model.vision;

import static org.junit.Assert.*;
import java.util.List;
import org.junit.Test;

public final class FlashNextMropeTest {

    @Test public void visionPositionsFollowReferencePattern() {
        // Grid (1,4,6), merge 2 -> merged grid 2 rows x 3 cols, start position 3.
        int[][] pos = FlashNextMrope.visionPositions(3, new ImageGrid(1, 4, 6), 2);
        assertArrayEquals(new int[]{3, 3, 3, 3, 3, 3}, pos[0]);
        assertArrayEquals(new int[]{3, 3, 3, 4, 4, 4}, pos[1]);
        assertArrayEquals(new int[]{3, 4, 5, 3, 4, 5}, pos[2]);
    }

    @Test public void ropeIndexAdvancesByMaxMergedExtentAndComputesDelta() {
        // 3 text, 6 image tokens (grid 1,4,6), 2 text. After the image the running
        // position advances by max(4,6)/2 = 3, so trailing text starts at 6, not 9.
        int imageToken = 99;
        int[] tokens = {1, 2, 3, imageToken, imageToken, imageToken, imageToken, imageToken, imageToken, 4, 5};
        var index = FlashNextMrope.ropeIndexForTokens(tokens, imageToken, List.of(new ImageGrid(1, 4, 6)), 2);
        assertArrayEquals(new int[]{0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7}, index.positions()[0]);
        assertArrayEquals(new int[]{0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7}, index.positions()[1]);
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7}, index.positions()[2]);
        assertEquals(7 + 1 - tokens.length, index.positionDelta());
    }

    @Test public void ropeIndexRejectsMismatchedExpansion() {
        int[] tokens = {99, 99, 99};
        assertThrows(IllegalArgumentException.class,
                () -> FlashNextMrope.ropeIndexForTokens(tokens, 99, List.of(new ImageGrid(1, 4, 6)), 2));
    }

    @Test public void tokenCountMatchesMergedGrid() {
        assertEquals(64, FlashNextMrope.tokenCount(new ImageGrid(1, 16, 16), 2));
        assertEquals(6, FlashNextMrope.tokenCount(new ImageGrid(1, 4, 6), 2));
        assertEquals(12, FlashNextMrope.tokenCount(new ImageGrid(2, 4, 6), 2));
        assertThrows(IllegalArgumentException.class,
                () -> FlashNextMrope.tokenCount(new ImageGrid(1, 3, 6), 2));
    }
}

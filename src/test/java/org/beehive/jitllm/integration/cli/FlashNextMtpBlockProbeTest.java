package org.beehive.jitllm.integration.cli;

import static org.junit.Assert.*;
import org.junit.Test;

public final class FlashNextMtpBlockProbeTest {
    @Test public void identicalReplayHasZeroError() {
        float[] values={1.5f,-2.25f,0,10.0625f};
        assertEquals(0,FlashNextMtpBlockProbe.maxAbsoluteError(values,values),0);
        assertTrue(FlashNextMtpBlockProbe.withinReplayTolerance(values,values));
    }
    @Test public void bf16RoundingStepAtTensorScaleIsAccepted() {
        // Observed staged-HC replay noise: one BF16 step (0.125) near magnitude 10.
        float[] reference={10.0625f,0.5f,-3f};
        float[] replay={9.9375f,0.5f,-3f};
        assertEquals(0.125,FlashNextMtpBlockProbe.maxAbsoluteError(replay,reference),0);
        assertTrue(FlashNextMtpBlockProbe.withinReplayTolerance(replay,reference));
    }
    @Test public void cancellationNoiseAtSmallElementsUsesTensorScale() {
        float[] reference={10f,0.01f};
        assertTrue(FlashNextMtpBlockProbe.withinReplayTolerance(new float[]{10f,0.09f},reference));
        assertFalse(FlashNextMtpBlockProbe.withinReplayTolerance(new float[]{10f,0.75f},reference));
    }
    @Test public void grossDivergenceAndWrongLengthAreRejected() {
        float[] reference={10f,1f,-1f};
        assertFalse(FlashNextMtpBlockProbe.withinReplayTolerance(new float[]{5f,1f,-1f},reference));
        assertFalse(FlashNextMtpBlockProbe.withinReplayTolerance(new float[]{10f,1f},reference));
    }
}

package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.*;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.*;

public final class FlashNextMtpKernelTest {
    @Test public void hiddenNormalizationUsesAllBranchesAndRepeatsWeightsAcrossRows() {
        var input=FloatArray.fromArray(new float[]{1,1,2,2,3,3,4,4,.5f,.5f,1,1,1.5f,1.5f,2,2});
        var output=new FloatArray(16);var weights=new FloatArray(8);
        FlashNextMtpKernels.gemmaNorm(input,weights,output,8,1e-6f);
        float[] expected={.365234375f,.365234375f,.73046875f,.73046875f,1.09375f,1.09375f,1.4609375f,1.4609375f};
        for(int i=0;i<16;i++) assertEquals(expected[i%8],output.get(i),0);
    }
    @Test public void embeddingProjectionBroadcastsWithinEachTokensBranches() {
        var embedding=new BFloat16Array(4);var hidden=new BFloat16Array(16);var output=new FloatArray(16);
        float[] values={1,2,4,8};for(int i=0;i<4;i++) embedding.set(i,(short)(Float.floatToRawIntBits(values[i])>>>16));
        FlashNextMtpKernels.fuse(embedding,hidden,output,2,4);
        for(int i=0;i<16;i++) assertEquals(values[i/8*2+i%2],output.get(i),0);
    }
}

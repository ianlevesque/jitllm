package org.beehive.jitllm.format.safetensors;

import static org.junit.Assert.*;
import org.junit.Test;

public final class Nvfp4ActivationTest {
    @Test public void e2m1UsesLowNibbleFirstAndRoundToNearestEven() {
        float[] values={0,.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2,2.5f,3,3.5f,4,5,6,-.75f};
        var result=Nvfp4Activation.quantize(values,1);
        assertArrayEquals(new byte[]{0,0x21,0x22,0x43,0x44,0x65,0x66,(byte)0xa7},result.values());
        assertArrayEquals(new byte[]{0x38},result.scales());
        assertArrayEquals(new float[]{0,0,.5f,1,1,1,1.5f,2,2,2,3,4,4,4,6,-1},result.dequantize(),0);
    }
    @Test public void zeroBlocksAreZeroAndMalformedInputsFail() {
        var result=Nvfp4Activation.quantize(new float[32],.001f);
        assertArrayEquals(new byte[16],result.values());assertArrayEquals(new byte[2],result.scales());
        assertThrows(IllegalArgumentException.class,()->Nvfp4Activation.quantize(new float[15],1));
        assertThrows(IllegalArgumentException.class,()->Nvfp4Activation.quantize(new float[16],0));
        float[] invalid=new float[16];invalid[0]=Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,()->Nvfp4Activation.quantize(invalid,1));
    }
}

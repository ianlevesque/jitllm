package org.beehive.jitllm.backend.tornado.kernels;

import org.junit.Test;
import static org.junit.Assert.*;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class FlashNextDenseKernelTest {
    @Test public void halfSubnormalsAndSignedValuesAreDecodedWithoutLosingExponentRange() {
        int[] bits={1,0x8001,0x3c00,0xbc00,0x7bff,0x0400};
        for(int value:bits) {
            var bytes=new ByteArray(2);bytes.set(0,(byte)value);bytes.set(1,(byte)(value>>>8));
            var input=new FloatArray(1);input.set(0,1);var output=new FloatArray(1);
            FlashNextWeightKernels.denseMatvec(input,bytes,output,1,1);
            assertEquals(Float.float16ToFloat((short)value),output.get(0),0f);
        }
    }
    @Test public void bfloatWeightsKeepLargeExponentsAndFloatWeightsKeepAllMantissaBits() {
        for(int type:new int[]{0,2}) {
            float value=type==0 ? Math.scalb(1.5f,80) : 1.234567f;
            int bits=Float.floatToRawIntBits(value);int width=type==0 ? 2 : 4;
            if(type==0) bits>>>=16;
            var bytes=new ByteArray(width);for(int i=0;i<width;i++) bytes.set(i,(byte)(bits>>>(i*8)));
            var input=new FloatArray(1);input.set(0,1);var output=new FloatArray(1);
            FlashNextWeightKernels.denseMatvec(input,bytes,output,1,type);
            assertEquals(value,output.get(0),0f);
        }
    }
}

package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.*;

public final class FlashNextMtpKernels {
    private FlashNextMtpKernels() {}
    /** Hidden width is all HC branches; norm weights repeat across batch rows only. */
    public static void gemmaNorm(FloatArray input,FloatArray weights,FloatArray output,int width,float epsilon) {
        for(@Parallel int row=0;row<input.getSize()/width;row++) {
            double sum=0;
            for(int j=0;j<width;j++) {float x=input.get(row*width+j);sum+=x*x;}
            float inverse=(float)(1/TornadoMath.sqrt((double)((float)(sum/width)+epsilon)));
            for(int j=0;j<width;j++) output.set(row*width+j,
                    FlashNextDeltaNetKernels.bf16(input.get(row*width+j)*inverse*(1+weights.get(j))));
        }
    }
    public static void fuse(BFloat16Array embedding,BFloat16Array hidden,FloatArray output,int width,int branches) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            int row=i/(width*branches),column=i%width;
            float a=Float.intBitsToFloat((embedding.get(row*width+column)&65535)<<16);
            float b=Float.intBitsToFloat((hidden.get(i)&65535)<<16);
            output.set(i,FlashNextDeltaNetKernels.bf16(a+b));
        }
    }
}

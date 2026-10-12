package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** E4M3 group encoding and scaled FP32 accumulation for checkpoint block-FP8 weights. */
public final class FlashNextFp8Kernels {
    private FlashNextFp8Kernels() {}
    public static void quantize(FloatArray input,FloatArray table,ByteArray encoded,FloatArray scales) {
        for(@Parallel int block=0;block<scales.getSize();block++) {
            float largest=1e-10f;
            for(int i=0;i<128;i++) largest=TornadoMath.max(largest,TornadoMath.abs(input.get(block*128+i)));
            float scale=largest*(1f/448f);scales.set(block,scale);
            float inverse=448f/largest;
            for(int i=0;i<128;i++) {
                float value=input.get(block*128+i)*inverse,absolute=TornadoMath.min(448f,TornadoMath.abs(value));
                int low=0,high=126;
                while(high-low>1) {int mid=(low+high)/2;if(table.get(mid)<=absolute) low=mid;else high=mid;}
                float a=absolute-table.get(low),b=table.get(high)-absolute;
                int code=a<b || (a==b && (low&1)==0) ? low : high;
                encoded.set(block*128+i,(byte)(code|((Float.floatToRawIntBits(value)>>>24)&128)));
            }
        }
    }
    public static void reduce(FloatArray partials,FloatArray weightScales,FloatArray inputScales,
            FloatArray control,FloatArray output,int rows,int columns) {
        int groups=columns/128,batch=output.getSize()/rows;
        for(@Parallel int i=0;i<output.getSize();i++) {
            int row=i%rows,token=i/rows;float sum=0;
            for(int group=0;group<groups;group++) {
                float scale=weightScales.get(row/128*groups+group)*inputScales.get(token*groups+group);
                sum+=partials.get(group*batch*rows+i)*scale;
            }
            output.set(i,FlashNextDeltaNetKernels.bf16(sum*control.get(3)));
        }
    }
    public static void seed(FloatArray table,FloatArray control,FloatArray scratch) {
        scratch.set(0,table.get(0)+control.get(2));
    }
    public static void readExpertBf16(BFloat16Array input,FloatArray output,int rows,int batch) {
        for(@Parallel int i=0;i<output.getSize();i++) output.set(i,Float.intBitsToFloat((input.get(i/rows*batch*rows+i%rows)&65535)<<16));
    }
    public static void readExpertFloat(FloatArray input,FloatArray output,int rows,int batch) {
        for(@Parallel int i=0;i<output.getSize();i++) output.set(i,input.get(i/rows*batch*rows+i%rows));
    }
    public static void reduceExperts(FloatArray partials,FloatArray weightScales,FloatArray inputScales,
            FloatArray routes,BFloat16Array output,int rows,int columns,int batch,int perExpertInput,int routed) {
        int groups=columns/128;
        for(@Parallel int i=0;i<output.getSize();i++) {
            int expert=i/(rows*batch),row=i%rows,token=i/rows%batch;float sum=0;
            for(int group=0;group<groups;group++) {
                float scale=weightScales.get((expert*(rows/128)+row/128)*groups+group)
                        *inputScales.get((perExpertInput*expert*batch+token)*groups+group);
                sum+=partials.get(((expert*groups+group)*batch+token)*rows+row)*scale;
            }
            float weighted=sum*(routed==1 ? routes.get(expert) : 1);
            output.set(i,(short)(Float.floatToRawIntBits(FlashNextDeltaNetKernels.bf16(weighted))>>>16));
        }
    }
    public static void finish(BFloat16Array down,FloatArray shared,FloatArray sharedRouter,FloatArray output,int count,int batch) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            float sum=0;for(int expert=0;expert<count;expert++) sum+=Float.intBitsToFloat((down.get(expert*batch*output.getSize()+i)&65535)<<16);
            float gate=FlashNextDeltaNetKernels.bf16((float)(1/(1+TornadoMath.exp(-(double)sharedRouter.get(0)))));
            output.set(i,FlashNextDeltaNetKernels.bf16(FlashNextDeltaNetKernels.bf16(sum)+FlashNextDeltaNetKernels.bf16(shared.get(i)*gate)));
        }
    }
}

package org.beehive.jitllm.format.safetensors;

/** Host block-FP8 W8A8 encoding with one dynamic FP32 scale per 128 values.
 * Uses IEEE division; CUDA's staged fast reciprocal can differ at rounding boundaries.
 */
public final class Fp8Activation {
    private Fp8Activation() {}
    public record Packed(byte[] values,float[] scales) {
        public Packed {
            if(values.length==0 || values.length%128!=0 || scales.length!=values.length/128)
                throw new IllegalArgumentException("Invalid block-FP8 activation");
            for(float scale:scales) if(!Float.isFinite(scale) || scale<=0) throw new IllegalArgumentException("Invalid FP8 scale");
        }
    }
    public static Packed quantize(float[] input) {
        if(input.length==0 || input.length%128!=0) throw new IllegalArgumentException("FP8 groups require 128 values");
        byte[] values=new byte[input.length];float[] scales=new float[input.length/128];
        for(int block=0;block<scales.length;block++) {
            float maximum=1e-10f;
            for(int i=0;i<128;i++) {float value=input[block*128+i];if(!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite FP8 activation");maximum=Math.max(maximum,Math.abs(value));}
            scales[block]=maximum*(1f/448f);float inverse=448f/maximum;
            for(int i=0;i<128;i++) {
                float value=input[block*128+i]*inverse,magnitude=Math.min(448f,Math.abs(value));int low=0,high=126;
                while(high-low>1) {int mid=(low+high)/2;if(FloatFormats.e4m3(mid)<=magnitude) low=mid;else high=mid;}
                float a=magnitude-FloatFormats.e4m3(low),b=FloatFormats.e4m3(high)-magnitude;
                int code=a<b || (a==b && (low&1)==0) ? low : high;
                values[block*128+i]=(byte)(code|((Float.floatToRawIntBits(value)>>>24)&128));
            }
        }
        return new Packed(values,scales);
    }
}

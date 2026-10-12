package org.beehive.jitllm.format.safetensors;

/** ModelOpt NVFP4 activation quantization: 16-element E4M3 blocks and E2M1 values. */
public final class Nvfp4Activation {
    public record Packed(byte[] values,byte[] scales,float globalScale) {
        public float[] dequantize() {
            float[] output=new float[values.length*2];
            for(int i=0; i<output.length; i++) {
                int nibble=((values[i/2]&255) >>> ((i&1)*4))&15;
                output[i]=FloatFormats.e2m1(nibble)*FloatFormats.e4m3(scales[i/16])*globalScale;
            }
            return output;
        }
    }
    private Nvfp4Activation() {}
    public static Packed quantize(float[] input,float globalScale) {
        if(input.length%16!=0 || !Float.isFinite(globalScale) || globalScale<=0)
            throw new IllegalArgumentException("Invalid NVFP4 activation dimensions/global scale");
        float inverseGlobal=1/globalScale;
        if(!Float.isFinite(inverseGlobal)) throw new IllegalArgumentException("NVFP4 global scale is too small");
        byte[] packed=new byte[input.length/2],scales=new byte[input.length/16];
        for(int start=0; start<input.length; start+=16) {
            float maximum=0;
            for(int i=start; i<start+16; i++) {
                if(!Float.isFinite(input[i])) throw new IllegalArgumentException("Nonfinite NVFP4 activation");
                maximum=Math.max(maximum,Math.abs(input[i]));
            }
            int encoded=e4m3(maximum*inverseGlobal/6);
            scales[start/16]=(byte)encoded;
            float decoded=FloatFormats.e4m3(encoded);
            float inverse=decoded==0 ? 0 : inverseGlobal/decoded;
            for(int i=start; i<start+16; i++) {
                int nibble=e2m1(input[i]*inverse);
                packed[i/2]|=(byte)(nibble<<((i&1)*4));
            }
        }
        return new Packed(packed,scales,globalScale);
    }
    public static int e4m3(float value) {
        if(Float.isNaN(value) || value<0) throw new IllegalArgumentException("Activation block scale must be nonnegative");
        if(value>=448) return 126;
        int lo=0,hi=126;
        while(lo<hi) {
            int middle=(lo+hi)/2;
            if(FloatFormats.e4m3(middle)<value) lo=middle+1; else hi=middle;
        }
        if(lo==0) return 0;
        float lower=value-FloatFormats.e4m3(lo-1),upper=FloatFormats.e4m3(lo)-value;
        return lower<upper || (lower==upper && ((lo-1)&1)==0) ? lo-1 : lo;
    }
    private static int e2m1(float value) {
        int sign=(Float.floatToRawIntBits(value)>>>28)&8;
        float magnitude=Math.abs(value);
        int best=0;
        float distance=Float.POSITIVE_INFINITY;
        for(int code=0; code<8; code++) {
            float candidate=Math.abs(magnitude-FloatFormats.e2m1(code));
            if(candidate<distance || (candidate==distance && (code&1)==0)) { best=code; distance=candidate; }
        }
        if(magnitude>=6) best=7;
        return sign|best;
    }
}

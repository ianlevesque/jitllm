package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Storage-preserving scalar projections and CUDA encoding helpers for native GEMM. */
public final class FlashNextWeightKernels {
    /** CUDA fast reciprocal encoding, matching the staged FlashInfer NVFP4 quantizer. */
    public static void quantizeNvfp4(FloatArray input,FloatArray calibration,IntArray mode,FloatArray e4m3,FloatArray e2m1,ByteArray packed,ByteArray scales) {
        for(@Parallel int block=0;block<scales.getSize();block++) {
            if(mode.get(1)==1) {
                float maximum=0;for(int i=0;i<16;i++) maximum=TornadoMath.max(maximum,TornadoMath.abs(input.get(block*16+i)));
                float inverseGlobal=calibration.get(2),sf=inverseGlobal*(maximum*(1f/6f));
                int encoded=0;float distance=Float.MAX_VALUE;
                // Runtime bounds prevent Graal from cloning the nested encoder
                // hundreds of times for every newly selected expert plan.
                int e4Codes=mode.get(2),e2Codes=mode.get(3),lanes=mode.get(4);
                for(int code=0;code<e4Codes;code++) {float d=TornadoMath.abs(sf-e4m3.get(code));if(d<distance || (d==distance && (code&1)==0)) {encoded=code;distance=d;}}
                scales.set(block,(byte)encoded);
                float inverse=maximum==0 ? 0 : 1f/(e4m3.get(encoded)*(1f/inverseGlobal));
                for(int pair=0;pair<e2Codes;pair++) {
                    int byteValue=0;
                    for(int lane=0;lane<lanes;lane++) {
                        float value=input.get(block*16+pair*2+lane)*inverse;int sign=(Float.floatToRawIntBits(value)>>>28)&8;
                        float magnitude=TornadoMath.abs(value),bestDistance=Float.MAX_VALUE;int best=0;
                        for(int code=0;code<e2Codes;code++) {float delta=TornadoMath.abs(magnitude-e2m1.get(code));if(delta<bestDistance || (delta==bestDistance && (code&1)==0)) {best=code;bestDistance=delta;}}
                        if(magnitude>=6) best=7;
                        byteValue|=(sign|best)<<(lane*4);
                    }
                    packed.set(block*8+pair,(byte)byteValue);
                }
            }
        }
    }
    /** cuBLASLt's 128x4 scale tile, retaining the original checkpoint's scale bytes. */
    public static void swizzleNvfp4Scales(ByteArray input,ByteArray output,IntArray initialized,int rows,int columns,int once) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            if(once==0 || initialized.get(0)==0) {
                int tile=i/512,local=i%512,tilesPerRow=columns/4;
                int row=(tile/tilesPerRow)*128+(local%16)/4*32+local/16;
                int column=(tile%tilesPerRow)*4+local%4;
                output.set(i,row<rows ? input.get(row*columns+column) : (byte)0);
            }
        }
    }
    private FlashNextWeightKernels() {}
    public static void bf16GemmInput(FloatArray input,BFloat16Array packed,int columns,int broadcast) {
        for(@Parallel int i=0;i<packed.getSize();i++) {
            float value=FlashNextDeltaNetKernels.bf16(input.get(broadcast==1 ? i%columns : i));
            packed.set(i,(short)(Float.floatToRawIntBits(value)>>>16));
        }
    }
    public static void bf16GemmOutput(BFloat16Array input,FloatArray output) {
        for(@Parallel int i=0;i<output.getSize();i++) output.set(i,Float.intBitsToFloat((input.get(i)&65535)<<16));
    }

    public static void denseMatvec(FloatArray input,ByteArray weights,FloatArray output,int columns,int dtype) {
        int width=dtype==2 ? 4 : 2;
        for(@Parallel int row=0;row<output.getSize();row++) {
            float sum=0;
            for(int column=0;column<columns;column++) {
                int offset=(row*columns+column)*width;
                int bits=(weights.get(offset)&255)|((weights.get(offset+1)&255)<<8);
                float weight;
                if(dtype==0) weight=Float.intBitsToFloat(bits<<16);
                else if(dtype==1) {
                    int exponent=(bits>>>10)&31,mantissa=bits&1023;
                    int normalized=mantissa;
                    int shift=0;
                    if(exponent==0 && mantissa!=0) {
                        while((normalized&1024)==0) { normalized<<=1;shift++; }
                        weight=Float.intBitsToFloat(((bits&32768)<<16)|((113-shift)<<23)|((normalized&1023)<<13));
                    } else weight=Float.intBitsToFloat(((bits&32768)<<16)|((exponent==31 ? 255 : exponent==0 ? 0 : exponent+112)<<23)|(mantissa<<13));
                } else weight=Float.intBitsToFloat(bits|((weights.get(offset+2)&255)<<16)|((weights.get(offset+3)&255)<<24));
                sum+=weight*input.get(column);
            }
            output.set(row,sum);
        }
    }

    /** Packed W4A4 execution; apply global weight/activation alpha after accumulation. */
    public static void nvfp4W4A4Matvec(ByteArray input,ByteArray inputScales,ByteArray weights,ByteArray scales,
                                      FloatArray e2m1,FloatArray e4m3,FloatArray alpha,FloatArray output,int columns) {
        int blocks=columns/16;
        for(@Parallel int row=0;row<output.getSize();row++) {
            float sum=0;
            for(int column=0;column<columns;column++) {
                int shift=(column&1)*4;
                float weight=e2m1.get(((weights.get(row*(columns/2)+column/2)&255)>>>shift)&15)*e4m3.get(scales.get(row*blocks+column/16)&255);
                float value=e2m1.get(((input.get(column/2)&255)>>>shift)&15)*e4m3.get(inputScales.get(column/16)&255);
                sum+=weight*value;
            }
            output.set(row,sum*alpha.get(0));
        }
    }

    /** One reusable graph accepts packed W4A4 or floating reference activations. */
    public static void nvfp4Projection(FloatArray input,ByteArray packed,ByteArray inputScales,IntArray mode,
                                      ByteArray weights,ByteArray scales,FloatArray e2m1,FloatArray e4m3,
                                      FloatArray alpha,FloatArray output,int columns) {
        int blocks=columns/16;
        for(@Parallel int row=0;row<output.getSize();row++) {
            float sum=0;
            for(int column=0;column<columns;column++) {
                int shift=(column&1)*4;
                float weight=e2m1.get(((weights.get(row*(columns/2)+column/2)&255)>>>shift)&15)*e4m3.get(scales.get(row*blocks+column/16)&255);
                float value=mode.get(0)==1
                        ? e2m1.get(((packed.get(column/2)&255)>>>shift)&15)*e4m3.get(inputScales.get(column/16)&255)
                        : input.get(column);
                sum+=weight*value;
            }
            output.set(row,sum*alpha.get(0));
        }
    }

    public static void nvfp4Matvec(FloatArray input, ByteArray weights, ByteArray scales,
                                   FloatArray e2m1, FloatArray e4m3, FloatArray globalScale,
                                   FloatArray output, int columns) {
        int blocks = columns / 16;
        for (@Parallel int row = 0; row < output.getSize(); row++) {
            float sum = 0;
            for (int column = 0; column < columns; column++) {
                int packed = weights.get(row * (columns / 2) + column / 2) & 255;
                int nibble = (packed >>> ((column & 1) * 4)) & 15;
                float scale = e4m3.get(scales.get(row * blocks + column / 16) & 255);
                float weight = e2m1.get(nibble) * scale * globalScale.get(0);
                sum += weight * input.get(column);
            }
            output.set(row, sum);
        }
    }

    public static void blockFp8Matvec(FloatArray input, ByteArray weights, FloatArray scales,
                                      FloatArray e4m3, FloatArray output, int columns) {
        int blocks = (columns + 127) / 128;
        for (@Parallel int row = 0; row < output.getSize(); row++) {
            float sum = 0;
            for (int column = 0; column < columns; column++) {
                float weight = e4m3.get(weights.get(row * columns + column) & 255)
                        * scales.get((row / 128) * blocks + column / 128);
                sum += weight * input.get(column);
            }
            output.set(row, sum);
        }
    }
}

package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;

/** Storage-preserving reference projections. Activations supplied by the caller are FP32.
 * W4A4 quantization and tensor-core dispatch must be added before using this in the serving plan.
 */
public final class FlashNextWeightKernels {
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

package org.beehive.jitllm.format.safetensors;

import java.io.IOException;

/** Optional executor for a validated checkpoint matrix, preserving its packed storage. */
public interface MatrixExecutor {
    void multiply(ModelOptMatrix matrix, float[] input, float[] output) throws IOException;
    void multiplyNvfp4(ModelOptMatrix matrix, Nvfp4Activation.Packed input, float[] output) throws IOException;
    default boolean quantizesNvfp4OnDevice() {return false;}
    default void multiplyNvfp4Values(ModelOptMatrix matrix,float[] input,float scale,float[] output) throws IOException {
        multiplyNvfp4(matrix,Nvfp4Activation.quantize(input,scale),output);
    }
    default void multiplyBf16(ModelOptMatrix matrix,float[] input,float[] output,int prefillRows) throws IOException {
        multiply(matrix,input,output);
        for(int i=0;i<output.length;i++) output[i]=FloatFormats.roundBf16(output[i]);
    }
}

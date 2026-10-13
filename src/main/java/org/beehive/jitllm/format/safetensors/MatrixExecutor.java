package org.beehive.jitllm.format.safetensors;

import java.io.IOException;

/** Optional executor for a validated checkpoint matrix, preserving its packed storage. */
public interface MatrixExecutor {
    void multiply(ModelOptMatrix matrix, float[] input, float[] output) throws IOException;
    void multiplyNvfp4(ModelOptMatrix matrix, Nvfp4Activation.Packed input, float[] output) throws IOException;
    /** BF16 operands with an FP32 accumulator/store; optional K partitions preserve HC decode precision. */
    default void multiplyBf16Float(ModelOptMatrix matrix,float[] input,float[] output,int partition) throws IOException {multiply(matrix,input,output);}
    default boolean quantizesNvfp4OnDevice() {return false;}
    default void multiplyNvfp4Values(ModelOptMatrix matrix,float[] input,float scale,float[] output) throws IOException {
        multiplyNvfp4(matrix,Nvfp4Activation.quantize(input,scale),output);
    }
    default void multiplyBf16(ModelOptMatrix matrix,float[] input,float[] output,int prefillRows) throws IOException {
        multiply(matrix,input,output);
        for(int i=0;i<output.length;i++) output[i]=FloatFormats.roundBf16(output[i]);
    }
    default void multiplyBf16Batch(ModelOptMatrix matrix,float[] input,float[] output,int rows) throws IOException {
        if(rows<=0 || input.length!=Math.multiplyExact(rows,matrix.columns()) || output.length!=Math.multiplyExact(rows,matrix.rows()))
            throw new IllegalArgumentException("Invalid BF16 batch");
        for(int row=0;row<rows;row++) {
            float[] values=java.util.Arrays.copyOfRange(input,row*matrix.columns(),(row+1)*matrix.columns()),result=new float[matrix.rows()];
            multiplyBf16(matrix,values,result,rows);System.arraycopy(result,0,output,row*matrix.rows(),result.length);
        }
    }
    default void multiplyBlockFp8Values(ModelOptMatrix matrix,float[] input,float routeWeight,float[] output) throws IOException {
        matrix.multiplyBlockFp8Reference(Fp8Activation.quantize(input),routeWeight,output);
    }
}

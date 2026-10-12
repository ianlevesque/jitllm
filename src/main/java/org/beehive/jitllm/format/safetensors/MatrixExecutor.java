package org.beehive.jitllm.format.safetensors;

import java.io.IOException;

/** Optional executor for a validated checkpoint matrix, preserving its packed storage. */
public interface MatrixExecutor {
    void multiply(ModelOptMatrix matrix, float[] input, float[] output) throws IOException;
    void multiplyNvfp4(ModelOptMatrix matrix, Nvfp4Activation.Packed input, float[] output) throws IOException;
}

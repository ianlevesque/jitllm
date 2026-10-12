package org.beehive.jitllm.inference.op;

import java.io.IOException;
import org.beehive.jitllm.format.safetensors.ModelOptMatrix;

/** Request-independent MoE layer resources; the implementation owns all transient activations. */
public interface MoeExecutor {
    record Result(float[] output,int[] experts,float[] weights) {}
    interface Layer extends AutoCloseable {
        Result forward(float[] input) throws IOException;
        default float[] routingWeights() throws IOException {throw new UnsupportedOperationException("No routing diagnostics");}
        default java.util.Map<String,float[]> intermediates() throws IOException {throw new UnsupportedOperationException("No MoE intermediate diagnostics");}
        @Override void close() throws IOException;
    }
    Layer newLayer(ModelOptMatrix router,ModelOptMatrix sharedGate,ModelOptMatrix sharedUp,ModelOptMatrix sharedDown,ModelOptMatrix sharedRouter,
                   ModelOptMatrix[] gates,ModelOptMatrix[] ups,ModelOptMatrix[] downs,int topK,boolean normalize,float inputScale,float intermediateScale) throws IOException;
}

package org.beehive.jitllm.inference.op;

import java.io.IOException;
import org.beehive.jitllm.format.safetensors.ModelOptMatrix;

/** Shared-branch MTP input fusion with global hidden Gemma normalization. */
public interface MtpFusionExecutor {
    interface Fusion extends AutoCloseable {
        float[] forward(float[] embeddings,float[] hidden,int rows) throws IOException;
        @Override void close() throws IOException;
    }
    Fusion newFusion(ModelOptMatrix embeddingProjection,ModelOptMatrix hiddenProjection,int branches,float epsilon,
                     float[] embeddingNorm,float[] hiddenNorm) throws IOException;
}

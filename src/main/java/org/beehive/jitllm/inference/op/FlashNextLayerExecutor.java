package org.beehive.jitllm.inference.op;

import java.io.IOException;
import org.beehive.jitllm.format.safetensors.ModelOptMatrix;

/** Backend-neutral HC arithmetic and request-owned PLE convolution; table lookup stays in the loader. */
public interface FlashNextLayerExecutor {
    record GroupedShape(int dimension,int branches,float epsilon) {
        public GroupedShape {
            if(dimension<=0 || branches<=0 || !Float.isFinite(epsilon) || epsilon<=0)
                throw new IllegalArgumentException("Invalid grouped layer geometry");
            Math.multiplyExact(dimension,branches);
        }
        public int width() {return dimension*branches;}
    }
    record PleShape(GroupedShape grouped,int kernel,int dilation) {
        public PleShape {
            java.util.Objects.requireNonNull(grouped);
            if(kernel<=0 || dilation<=0) throw new IllegalArgumentException("Invalid PLE convolution geometry");
            Math.multiplyExact(grouped.width(),Math.multiplyExact(kernel-1,dilation));
        }
        public int history() {return (kernel-1)*dilation;}
        public int stateSize() {return grouped.width()*history();}
    }
    interface HyperConnection extends AutoCloseable {
        float[] normalize(float[] hidden) throws IOException;
        float[] activateLow(float[] projected,boolean fused) throws IOException;
        float[] mix(float[] normalized,float[] projectedGates,boolean fused) throws IOException;
        default boolean projectsMix() {return false;}
        default float[] mixNormalized(float[] normalized,java.util.function.BiConsumer<String,float[]> trace) throws IOException {throw new UnsupportedOperationException("Native HC mix projection is unavailable");}
        float[] combine(float[] block,float[] residual,float[] projectedInjection) throws IOException;
        default boolean projectsInjection() {return false;}
        default float[] combineNormalized(float[] block,float[] residual,float[] normalized) throws IOException {throw new UnsupportedOperationException("Native HC injection is unavailable");}
        @Override void close() throws IOException;
    }
    interface Snapshot extends AutoCloseable {@Override void close();}
    interface PleSession extends AutoCloseable {
        /** BF16-valued inputs; optionally add the PLE result to the residual on the device. */
        float[] forward(float[] hidden,float[] projectedKey,float[] projectedValue,boolean fusedDecode,boolean addResidual) throws IOException;
        Snapshot snapshot() throws IOException;
        void restore(Snapshot snapshot) throws IOException;
        void reset() throws IOException;
        @Override void close() throws IOException;
    }
    HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization);
    default HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization,float[] injectionWeights) {return newHyperConnection(shape,lowrank,normalization);}
    default HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization,float[] injectionWeights,ModelOptMatrix down,ModelOptMatrix up) throws IOException {
        return newHyperConnection(shape,lowrank,normalization,injectionWeights);
    }
    PleSession newPleSession(PleShape shape,float[] keyNorm,float[] queryNorm,float[] convolutionNorm,float[] convolution);
}

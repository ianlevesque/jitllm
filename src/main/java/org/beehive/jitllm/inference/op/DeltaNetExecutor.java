package org.beehive.jitllm.inference.op;

import java.io.IOException;

/** Backend-neutral execution of causal convolution and a gated delta recurrence.
 * Sessions own mutable sequence state; snapshots must be released by their caller.
 */
public interface DeltaNetExecutor {
    record Shape(int keyHeads,int valueHeads,int keyDim,int valueDim,int convolutionKernel,float epsilon) {
        public Shape {
            if(keyHeads<=0 || valueHeads<=0 || valueHeads%keyHeads!=0 || keyDim<=0 || valueDim<=0
                    || convolutionKernel<=0 || !Float.isFinite(epsilon) || epsilon<=0)
                throw new IllegalArgumentException("Invalid DeltaNet geometry");
            Math.addExact(Math.multiplyExact(2,Math.multiplyExact(keyHeads,keyDim)),Math.multiplyExact(valueHeads,valueDim));
        }
        public int channels() {return 2*keyHeads*keyDim+valueHeads*valueDim;}
        public int convolutionSize() {return Math.multiplyExact(channels(),convolutionKernel-1);}
        public int temporalSize() {return Math.multiplyExact(Math.multiplyExact(valueHeads,keyDim),valueDim);}
        public int stateSize() {return Math.addExact(convolutionSize(),temporalSize());}
    }
    interface Snapshot extends AutoCloseable {
        @Override void close();
    }
    interface Session extends AutoCloseable {
        /** BF16-valued projections in, BF16 gated/RMS-normalized recurrent output out. */
        float[] forward(float[] qkv,float[] gate,float[] alpha,float[] beta) throws IOException;
        Snapshot snapshot() throws IOException;
        void restore(Snapshot snapshot) throws IOException;
        void reset() throws IOException;
        @Override void close() throws IOException;
    }
    Session newSession(Shape shape,float[] convolution,float[] normalization,float[] aLog,float[] dtBias);
}

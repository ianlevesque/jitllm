package org.beehive.jitllm.inference.op;

import java.io.IOException;

/** Backend-neutral compressed sparse attention with request-owned physical KV storage. */
public interface QsaExecutor {
    record Shape(int context,int heads,int kvHeads,int dimension,int indexHeads,int indexDimension,int ratio,int budget,int rotaryDimensions,float epsilon) {
        public Shape {
            if(context<=0 || heads<=0 || kvHeads<=0 || heads%kvHeads!=0 || dimension<=0 || indexHeads<=0 || indexDimension<=0
                    || ratio<2 || budget<ratio || budget%ratio!=0 || budget/ratio>512 || rotaryDimensions<0 || rotaryDimensions%2!=0
                    || rotaryDimensions>Math.min(dimension,indexDimension) || !Float.isFinite(epsilon) || epsilon<=0)
                throw new IllegalArgumentException("Invalid QSA geometry");
            Math.multiplyExact(heads,dimension);Math.multiplyExact(kvHeads,dimension);
        }
        public int kvWidth() {return kvHeads*dimension;}
        public int queryWidth() {return heads*dimension;}
        public int indexQueryWidth() {return indexHeads*indexDimension;}
        public int pageTokens() {return ratio*512;}
        public int selectedCapacity() {return budget+ratio-1;}
    }
    interface Snapshot extends AutoCloseable {@Override void close();}
    interface Session extends AutoCloseable {
        int tokens();
        /** BF16-valued projection inputs; gated BF16 head outputs before the output projection. */
        float[] forward(float[] qAndGate,float[] key,float[] value,float[] index,int[] positions,boolean prefill,boolean fresh,int prefillBlock) throws IOException;
        int[] selectedTokens() throws IOException;
        Snapshot snapshot() throws IOException;
        void restore(Snapshot snapshot) throws IOException;
        void reset() throws IOException;
        void finishFreshPrefill();
        @Override void close() throws IOException;
    }
    Session newSession(Shape shape,float[] qNorm,float[] kNorm,float[] indexQNorm,float[] indexKNorm,int[] ropeAxes,float[] frequencies,boolean fp8,boolean fused);
}

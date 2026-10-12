package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.loader.Qwen4PleTable;
import org.beehive.jitllm.model.qwen4.PleHash;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;

/** BF16 host reference for Qwen4 hyper-connections and stateful staged PLE. */
public final class Qwen4CpuLayers {
    private Qwen4CpuLayers() {}
    private static float bf(float value) { return FloatFormats.roundBf16(value); }
    private static float sigmoid(float value) { return bf((float)(1/(1+Math.exp(-value)))); }
    private static float silu(float value) { return bf((float)(value/(1+Math.exp(-value)))); }

    public static float[] vector(SafeTensorIndex index, PagedTensorReader reader, String name, int length) throws IOException {
        var tensor=index.require(name);
        if(tensor.elements()!=length || !(tensor.dtype().equals("BF16") || tensor.dtype().equals("F32")))
            throw new IOException("Invalid vector: "+name);
        ByteBuffer bytes=ByteBuffer.wrap(reader.read(tensor,0,Math.toIntExact(tensor.bytes()))).order(ByteOrder.LITTLE_ENDIAN);
        float[] values=new float[length];
        for(int i=0; i<length; i++) values[i]=tensor.dtype().equals("BF16") ? FloatFormats.bf16(bytes.getShort()) : bytes.getFloat();
        return values;
    }

    static ModelOptMatrix matrix(SafeTensorIndex index,PagedTensorReader reader,String prefix,int rows,int columns) throws IOException {
        return matrix(index,reader,prefix,rows,columns,null);
    }
    static ModelOptMatrix matrix(SafeTensorIndex index,PagedTensorReader reader,String prefix,int rows,int columns,MatrixExecutor executor) throws IOException {
        var matrix=new ModelOptMatrix(index,reader,prefix,executor);
        if(matrix.rows()!=rows || matrix.columns()!=columns) throw new IOException("Invalid projection dimensions: "+prefix);
        return matrix;
    }
    static float[] project(ModelOptMatrix matrix,float[] input) throws IOException {
        float[] output=new float[matrix.rows()];
        matrix.multiply(input,output);
        for(int i=0; i<output.length; i++) output[i]=bf(output[i]);
        return output;
    }
    private static float[] norm(float[] input,float[] weights,int group,float epsilon) {
        if(input.length!=weights.length || input.length%group!=0) throw new IllegalArgumentException("Invalid grouped norm shape");
        float[] output=new float[input.length];
        for(int start=0; start<input.length; start+=group) {
            double sum=0;
            for(int j=0; j<group; j++) sum+=input[start+j]*input[start+j];
            float variance=(float)(sum/group);
            float factor=(float)(1/Math.sqrt(variance+epsilon));
            for(int j=0; j<group; j++) output[start+j]=bf(input[start+j]*factor*(1+weights[start+j]));
        }
        return output;
    }

    public static final class HyperConnection {
        public record Mixed(float[] output,float[] residual,float[] normalized) {}
        private final Qwen4Configuration config;
        private final float[] norm;
        private final ModelOptMatrix down,up,inject;
        public HyperConnection(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,boolean combine) throws IOException {
            this(config,index,reader,prefix,combine,null);
        }
        public HyperConnection(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,boolean combine,MatrixExecutor executor) throws IOException {
            this.config=config;
            norm=vector(index,reader,prefix+".hc_norm.weight",config.hyperDim());
            down=matrix(index,reader,prefix+".input_mix_weight_down",config.hcLowrank(),config.hyperDim(),executor);
            up=matrix(index,reader,prefix+".input_mix_weight_up",config.hyperDim(),config.hcLowrank(),executor);
            inject=combine ? matrix(index,reader,prefix+".block_inject_weight",config.hcCount(),config.hyperDim(),executor) : null;
        }
        public Mixed mix(float[] input) throws IOException {
            if(input.length!=config.hyperDim()) throw new IllegalArgumentException("Wrong hyper-connection input width");
            float[] normalized=norm(input,norm,config.dim(),config.rmsNormEps());
            // The staged SM120 persistent HC kernel retains both GEMM accumulators
            // in FP32. Its low-rank SiLU is the only intermediate BF16 store.
            float[] low=new float[down.rows()];down.multiply(normalized,low);
            for(int i=0; i<low.length; i++) low[i]=silu(low[i]/config.hcCount());
            float[] gates=new float[up.rows()],output=new float[config.dim()];up.multiply(low,gates);
            for(int j=0; j<output.length; j++) {
                float sum=0;
                for(int branch=0; branch<config.hcCount(); branch++) {
                    int offset=branch*config.dim()+j;
                    float weight=(float)(1/(1+Math.exp(-gates[offset])));
                    sum+=weight*normalized[offset];
                }
                output[j]=bf(sum/config.hcCount());
            }
            return new Mixed(output,input.clone(),normalized);
        }
        public float[] combine(float[] block,Mixed mixed) throws IOException {
            if(inject==null || block.length!=config.dim()) throw new IllegalArgumentException("Hyper-connection combine is unavailable or has wrong width");
            float[] weights=new float[inject.rows()],output=new float[config.hyperDim()];inject.multiply(mixed.normalized(),weights);
            for(int branch=0; branch<config.hcCount(); branch++) {
                float weight=(float)(2/(1+Math.exp(-weights[branch]/config.hcCount())));
                for(int j=0; j<config.dim(); j++) {
                    int offset=branch*config.dim()+j;
                    output[offset]=bf(Math.fma(block[j],weight,mixed.residual()[offset]));
                }
            }
            return output;
        }
    }

    public static final class Ple {
        public final class State {
            private final PleHash.History history=table.hash().new History();
            private final float[] convolution=new float[config.hyperDim()*stateLength];
            public record Snapshot(int[] history,float[] convolution) {}
            public void reset() { history.reset(); Arrays.fill(convolution,0); }
            public Snapshot snapshot() { return new Snapshot(history.snapshot(),convolution.clone()); }
            public void restore(Snapshot state) {
                if(state.convolution().length!=convolution.length) throw new IllegalArgumentException("Wrong PLE state shape");
                history.restore(state.history());
                System.arraycopy(state.convolution(),0,convolution,0,convolution.length);
            }
        }
        private final Qwen4Configuration config;
        private final Qwen4PleTable table;
        private final ModelOptMatrix key,value;
        private final float[] normKey,normQuery,normConv,conv;
        private final int stateLength;
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix) throws IOException {
            this(config,index,reader,prefix,null);
        }
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,MatrixExecutor executor) throws IOException {
            this.config=config;
            table=new Qwen4PleTable(index,reader,prefix+".ple_embedding",config.eosToken());
            if(table.width()!=config.pleRowWidth() || table.hash().heads()!=config.pleHeads()) throw new IOException("PLE layout/config mismatch");
            key=matrix(index,reader,prefix+".key_proj",config.hyperDim(),config.pleEmbedDim(),executor);
            value=matrix(index,reader,prefix+".value_proj",config.dim(),config.pleEmbedDim(),executor);
            normKey=vector(index,reader,prefix+".norm_key.weight",config.hyperDim());
            normQuery=vector(index,reader,prefix+".norm_query.weight",config.hyperDim());
            normConv=vector(index,reader,prefix+".norm_conv.weight",config.hyperDim());
            conv=vector(index,reader,prefix+".conv1d.weight",config.hyperDim()*config.pleConvKernel());
            stateLength=(config.pleConvKernel()-1)*config.ngramSize();
        }
        public State newState() { return new State(); }
        public float[] forward(int token,float[] hidden,State state) throws IOException {
            if(token<0 || token>=config.vocabularySize() || hidden.length!=config.hyperDim()) throw new IllegalArgumentException("Invalid PLE input");
            float[] embeddings=new float[config.pleEmbedDim()];
            table.gather(state.history.push(token),embeddings);
            float[] keys=norm(project(key,embeddings),normKey,config.dim(),config.rmsNormEps());
            float[] query=norm(hidden,normQuery,config.dim(),config.rmsNormEps());
            float[] values=project(value,embeddings),gated=new float[config.hyperDim()];
            for(int branch=0; branch<config.hcCount(); branch++) {
                float dot=0;
                for(int j=0; j<config.dim(); j++) dot+=bf(keys[branch*config.dim()+j]*query[branch*config.dim()+j]);
                float gate=bf(bf(dot)/(float)Math.sqrt(config.dim()));
                float transformed=bf((float)Math.sqrt(Math.max(Math.abs(gate),bf(1e-6f))));
                transformed=bf(transformed*Math.signum(gate));
                float beta=sigmoid(transformed);
                for(int j=0; j<config.dim(); j++) gated[branch*config.dim()+j]=bf(beta*values[j]);
            }
            float[] normalized=norm(gated,normConv,config.dim(),config.rmsNormEps());
            float[] output=new float[config.hyperDim()];
            for(int channel=0; channel<output.length; channel++) {
                float sum=0;
                int start=channel*stateLength;
                for(int tap=0; tap<config.pleConvKernel()-1; tap++)
                    sum=Math.fma(state.convolution[start+tap*config.ngramSize()],conv[channel*config.pleConvKernel()+tap],sum);
                sum=Math.fma(normalized[channel],conv[channel*config.pleConvKernel()+config.pleConvKernel()-1],sum);
                output[channel]=bf(gated[channel]+silu(bf(sum)));
                if(stateLength>0) {
                    System.arraycopy(state.convolution,start+1,state.convolution,start,stateLength-1);
                    state.convolution[start+stateLength-1]=normalized[channel];
                }
            }
            return output;
        }
    }
}

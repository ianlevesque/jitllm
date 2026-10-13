package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.loader.Qwen4PleTable;
import org.beehive.jitllm.model.qwen4.PleHash;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor;

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

    public static final class HyperConnection implements AutoCloseable {
        public record Mixed(float[] output,float[] residual,float[] normalized) {}
        private final Qwen4Configuration config;
        private final float[] norm;
        private final ModelOptMatrix down,up,inject;
        private final FlashNextLayerExecutor.HyperConnection device;
        private boolean closed;
        public HyperConnection(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,boolean combine) throws IOException {
            this(config,index,reader,prefix,combine,null);
        }
        public HyperConnection(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,boolean combine,MatrixExecutor executor) throws IOException {
            this(config,index,reader,prefix,combine,executor,null);
        }
        public HyperConnection(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,boolean combine,MatrixExecutor executor,FlashNextLayerExecutor layerExecutor) throws IOException {
            this.config=config;
            norm=vector(index,reader,prefix+".hc_norm.weight",config.hyperDim());
            down=matrix(index,reader,prefix+".input_mix_weight_down",config.hcLowrank(),config.hyperDim(),executor);
            up=matrix(index,reader,prefix+".input_mix_weight_up",config.hyperDim(),config.hcLowrank(),executor);
            inject=combine ? matrix(index,reader,prefix+".block_inject_weight",config.hcCount(),config.hyperDim(),executor) : null;
            float[] injectionWeights=null;
            if(layerExecutor!=null && inject!=null && inject.dtype().equals("BF16") && config.dim()%8==0 && config.hyperDim()%2048==0) {
                injectionWeights=new float[Math.multiplyExact(inject.rows(),inject.columns())];var row=new float[inject.columns()];
                for(int branch=0;branch<inject.rows();branch++) {inject.readRow(branch,row);System.arraycopy(row,0,injectionWeights,branch*row.length,row.length);}
            }
            device=layerExecutor==null ? null : layerExecutor.newHyperConnection(new FlashNextLayerExecutor.GroupedShape(config.dim(),config.hcCount(),config.rmsNormEps()),config.hcLowrank(),norm,injectionWeights);
        }
        public Mixed mix(float[] input) throws IOException {
            return mix(input,1);
        }
        public Mixed mix(float[] input,int prefillRows) throws IOException {
            return mix(input,prefillRows,null);
        }
        public Mixed mix(float[] input,int prefillRows,java.util.function.BiConsumer<String,float[]> trace) throws IOException {
            if(closed) throw new IllegalStateException("Hyper-connection is closed");
            if(input.length!=config.hyperDim() || prefillRows<=0) throw new IllegalArgumentException("Wrong hyper-connection input width/prefill rows");
            boolean fused=prefillRows<=16;
            float[] normalized=device==null ? norm(input,norm,config.dim(),config.rmsNormEps()) : device.normalize(input);
            // SM120 uses the persistent F32-accumulator mix for <=16 rows. Larger
            // prefill chunks use torch.compile, with a BF16 store after each GEMM.
            float[] low=new float[down.rows()];
            if(fused) down.multiplyBf16Float(normalized,low,down.columns()%256==0 ? 256 : down.columns());else down.multiplyBf16(normalized,low,prefillRows);
            if(trace!=null) trace.accept("down_raw",low.clone());
            if(device!=null) low=device.activateLow(low,fused);
            else for(int i=0; i<low.length; i++) low[i]=silu((fused ? low[i] : bf(low[i]))/config.hcCount());
            if(trace!=null) trace.accept("low",low.clone());
            float[] gates=new float[up.rows()],output=new float[config.dim()];
            if(fused) up.multiplyBf16Float(low,gates,up.columns());else up.multiplyBf16(low,gates,prefillRows);
            if(trace!=null) trace.accept("up_raw",gates.clone());
            if(device!=null) return new Mixed(device.mix(normalized,gates,fused),input.clone(),normalized);
            for(int j=0; j<output.length; j++) {
                float sum=0;
                for(int branch=0; branch<config.hcCount(); branch++) {
                    int offset=branch*config.dim()+j;
                    float gate=fused ? gates[offset] : bf(gates[offset]);
                    float weight=(float)(1/(1+Math.exp(-gate)));
                    sum+=weight*normalized[offset];
                }
                output[j]=bf(sum/config.hcCount());
            }
            return new Mixed(output,input.clone(),normalized);
        }
        public float[] combine(float[] block,Mixed mixed) throws IOException {
            if(closed) throw new IllegalStateException("Hyper-connection is closed");
            if(inject==null || block.length!=config.dim()) throw new IllegalArgumentException("Hyper-connection combine is unavailable or has wrong width");
            if(device!=null && device.projectsInjection()) return device.combineNormalized(block,mixed.residual(),mixed.normalized());
            float[] weights=new float[inject.rows()],output=new float[config.hyperDim()];inject.multiply(mixed.normalized(),weights);
            if(device!=null) return device.combine(block,mixed.residual(),weights);
            for(int branch=0; branch<config.hcCount(); branch++) {
                float weight=(float)(2/(1+Math.exp(-weights[branch]/config.hcCount())));
                for(int j=0; j<config.dim(); j++) {
                    int offset=branch*config.dim()+j;
                    output[offset]=bf(Math.fma(block[j],weight,mixed.residual()[offset]));
                }
            }
            return output;
        }
        @Override public void close() throws IOException {if(!closed) {closed=true;if(device!=null) device.close();}}
    }

    public static final class Ple {
        public final class State implements AutoCloseable {
            private final Ple owner=Ple.this;
            private final PleHash.History history=table.hash().new History();
            private final float[] convolution;
            private final FlashNextLayerExecutor.PleSession device;
            private boolean closed;
            private State() {
                device=layerExecutor==null ? null : layerExecutor.newPleSession(new FlashNextLayerExecutor.PleShape(
                        new FlashNextLayerExecutor.GroupedShape(config.dim(),config.hcCount(),config.rmsNormEps()),config.pleConvKernel(),config.ngramSize()),normKey,normQuery,normConv,conv);
                convolution=new float[device==null ? config.hyperDim()*stateLength : 0];
            }
            public record Snapshot(State owner,int[] history,float[] convolution,FlashNextLayerExecutor.Snapshot device) implements AutoCloseable {
                @Override public void close() {if(device!=null) device.close();}
            }
            private void requireOpen() {if(closed) throw new IllegalStateException("PLE state is closed");}
            public void reset() {
                requireOpen();
                if(device!=null) try {device.reset();} catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}
                else Arrays.fill(convolution,0);
                history.reset();
            }
            public Snapshot snapshot() throws IOException {requireOpen();return new Snapshot(this,history.snapshot(),convolution.clone(),device==null ? null : device.snapshot());}
            public void restore(Snapshot saved) throws IOException {
                requireOpen();
                if(saved.owner()!=this || saved.convolution().length!=convolution.length) throw new IllegalArgumentException("Wrong PLE state owner/shape");
                if(device!=null) device.restore(saved.device());
                else System.arraycopy(saved.convolution(),0,convolution,0,convolution.length);
                history.restore(saved.history());
            }
            @Override public void close() throws IOException {if(!closed) {closed=true;if(device!=null) device.close();}}
        }
        private final Qwen4Configuration config;
        private final Qwen4PleTable table;
        private final ModelOptMatrix key,value;
        private final float[] normKey,normQuery,normConv,conv;
        private final int stateLength;
        private final FlashNextLayerExecutor layerExecutor;
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix) throws IOException {
            this(config,index,reader,prefix,null);
        }
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,MatrixExecutor executor) throws IOException {
            this(config,index,reader,prefix,executor,null);
        }
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,MatrixExecutor executor,FlashNextLayerExecutor layerExecutor) throws IOException {
            this.config=config;
            this.layerExecutor=layerExecutor;
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
            return forward(token,hidden,state,false,false);
        }
        public float[] forward(int token,float[] hidden,State state,boolean fusedDecode) throws IOException {
            return forward(token,hidden,state,fusedDecode,false);
        }
        public float[] apply(int token,float[] hidden,State state,boolean fusedDecode) throws IOException {
            return forward(token,hidden,state,fusedDecode,true);
        }
        private float[] forward(int token,float[] hidden,State state,boolean fusedDecode,boolean addResidual) throws IOException {
            state.requireOpen();
            if(state.owner!=this || token<0 || token>=config.vocabularySize() || hidden.length!=config.hyperDim()) throw new IllegalArgumentException("Invalid PLE input/state");
            float[] embeddings=new float[config.pleEmbedDim()];
            table.gather(state.history.push(token),embeddings);
            float[] projectedKey=project(key,embeddings),values=project(value,embeddings);
            if(state.device!=null) return state.device.forward(hidden,projectedKey,values,fusedDecode,addResidual);
            float[] keys=norm(projectedKey,normKey,config.dim(),config.rmsNormEps());
            float[] query=norm(hidden,normQuery,config.dim(),config.rmsNormEps());
            float[] gated=new float[config.hyperDim()];
            for(int branch=0; branch<config.hcCount(); branch++) {
                float dot=0;
                for(int j=0; j<config.dim(); j++) dot+=bf(keys[branch*config.dim()+j]*query[branch*config.dim()+j]);
                float gate=bf(bf(dot)/(float)Math.sqrt(config.dim()));
                float transformed=bf((float)Math.sqrt(Math.max(Math.abs(gate),fusedDecode ? 1e-6f : bf(1e-6f))));
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
                if(addResidual) output[channel]=bf(hidden[channel]+output[channel]);
                if(stateLength>0) {
                    System.arraycopy(state.convolution,start+1,state.convolution,start,stateLength-1);
                    state.convolution[start+stateLength-1]=normalized[channel];
                }
            }
            return output;
        }
    }
}

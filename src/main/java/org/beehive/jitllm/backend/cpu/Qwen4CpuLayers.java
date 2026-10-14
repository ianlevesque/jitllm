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
    /**
     * Distinct-row batched {@link #project}: {@code inputs} holds {@code rows} consecutive
     * rows of {@code matrix.columns()} values and the result holds {@code rows} consecutive
     * BF16-rounded rows of {@code matrix.rows()} values. On the host reference this is
     * bit-identical to projecting each row separately (the batch fallback rounds each row
     * exactly as {@code project} does); on a GEMM executor the batch geometry may round
     * differently in the low bits, as any prefill geometry change does.
     */
    static float[] projectBatch(ModelOptMatrix matrix,float[] inputs,int rows) throws IOException {
        float[] output=new float[Math.multiplyExact(rows,matrix.rows())];
        matrix.multiplyBf16Batch(inputs,output,rows);
        for(int i=0; i<output.length; i++) output[i]=bf(output[i]);
        return output;
    }
    static float[] flatten(float[][] rows,int width) {
        float[] flat=new float[Math.multiplyExact(rows.length,width)];
        for(int row=0; row<rows.length; row++) {
            if(rows[row].length!=width) throw new IllegalArgumentException("Wrong batched row width");
            System.arraycopy(rows[row],0,flat,row*width,width);
        }
        return flat;
    }
    static float[][] unflatten(float[] flat,int rows,int width) {
        if(flat.length!=Math.multiplyExact(rows,width)) throw new IllegalArgumentException("Wrong batched flat width");
        float[][] values=new float[rows][];
        for(int row=0; row<rows; row++) values[row]=Arrays.copyOfRange(flat,row*width,(row+1)*width);
        return values;
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
            device=layerExecutor==null ? null : layerExecutor.newHyperConnection(new FlashNextLayerExecutor.GroupedShape(config.dim(),config.hcCount(),config.rmsNormEps()),config.hcLowrank(),norm,injectionWeights,down,up);
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
            if(fused && device!=null && device.projectsMix()) return new Mixed(device.mixNormalized(normalized,trace),input.clone(),normalized);
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
            return new Mixed(mixHost(normalized,gates,fused),input.clone(),normalized);
        }
        private float[] mixHost(float[] normalized,float[] gates,boolean fused) {
            float[] output=new float[config.dim()];
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
            return output;
        }
        /**
         * Distinct-row batched {@link #mix} for non-fused prefill geometry ({@code
         * prefillRows > 16}): the grouped norm, low-rank activation and gate mix stay
         * per row (on the device session when present), while the down/up projections run
         * as one distinct-row GEMM each. Every row's arithmetic mirrors the non-fused
         * branch of {@code mix}; only the GEMM geometry changes. Fused geometry keeps the
         * FP32-accumulator path and must use {@code mix} per row.
         */
        public Mixed[] mixBatch(float[][] inputs,int prefillRows) throws IOException {
            if(closed) throw new IllegalStateException("Hyper-connection is closed");
            if(inputs.length==0 || prefillRows<=16) throw new IllegalArgumentException("Batched HC mix requires non-fused prefill geometry and at least one row");
            int rows=inputs.length;
            float[][] normalized=new float[rows][];
            for(int row=0; row<rows; row++) {
                if(inputs[row].length!=config.hyperDim()) throw new IllegalArgumentException("Wrong hyper-connection input width");
                normalized[row]=device==null ? norm(inputs[row],norm,config.dim(),config.rmsNormEps()) : device.normalize(inputs[row]);
            }
            float[] lowFlat=new float[Math.multiplyExact(rows,down.rows())];
            down.multiplyBf16Batch(flatten(normalized,config.hyperDim()),lowFlat,rows);
            float[][] activated=unflatten(lowFlat,rows,down.rows());
            for(int row=0; row<rows; row++) {
                if(device!=null) activated[row]=device.activateLow(activated[row],false);
                else for(int i=0; i<activated[row].length; i++) activated[row][i]=silu(bf(activated[row][i])/config.hcCount());
            }
            float[] gatesFlat=new float[Math.multiplyExact(rows,up.rows())];
            up.multiplyBf16Batch(flatten(activated,down.rows()),gatesFlat,rows);
            float[][] gates=unflatten(gatesFlat,rows,up.rows());
            Mixed[] mixed=new Mixed[rows];
            for(int row=0; row<rows; row++)
                mixed[row]=new Mixed(device!=null ? device.mix(normalized[row],gates[row],false) : mixHost(normalized[row],gates[row],false),inputs[row].clone(),normalized[row]);
            return mixed;
        }
        /** Batched {@link #combine}: the injection GEMV is tiny and its result feeds the
         * sigmoid unrounded, so rows keep the exact single-row path. */
        public float[][] combineBatch(float[][] blocks,Mixed[] mixed) throws IOException {
            if(blocks.length!=mixed.length) throw new IllegalArgumentException("Wrong batched combine shape");
            float[][] combined=new float[blocks.length][];
            for(int row=0; row<blocks.length; row++) combined[row]=combine(blocks[row],mixed[row]);
            return combined;
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

    /** Close closes the PLE table's dedicated page-cache reader when it owns one. */
    public static final class Ple implements AutoCloseable {
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
            this(config,index,reader,null,prefix,executor,layerExecutor);
        }
        /** {@code pleReader} (when non-null) pages PLE rows for every layer sharing it; when null, {@code jitllm.qwen4.plePageCacheBytes} selects a dedicated reader (default) or the shared {@code reader} (value 0). */
        public Ple(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,PagedTensorReader pleReader,String prefix,MatrixExecutor executor,FlashNextLayerExecutor layerExecutor) throws IOException {
            this.config=config;
            this.layerExecutor=layerExecutor;
            table=createTable(index,reader,pleReader,prefix+".ple_embedding",config.eosToken());
            if(table.width()!=config.pleRowWidth() || table.hash().heads()!=config.pleHeads()) throw new IOException("PLE layout/config mismatch");
            key=matrix(index,reader,prefix+".key_proj",config.hyperDim(),config.pleEmbedDim(),executor);
            value=matrix(index,reader,prefix+".value_proj",config.dim(),config.pleEmbedDim(),executor);
            normKey=vector(index,reader,prefix+".norm_key.weight",config.hyperDim());
            normQuery=vector(index,reader,prefix+".norm_query.weight",config.hyperDim());
            normConv=vector(index,reader,prefix+".norm_conv.weight",config.hyperDim());
            conv=vector(index,reader,prefix+".conv1d.weight",config.hyperDim()*config.pleConvKernel());
            stateLength=(config.pleConvKernel()-1)*config.ngramSize();
        }
        private static Qwen4PleTable createTable(SafeTensorIndex index,PagedTensorReader reader,PagedTensorReader pleReader,String prefix,int eos) throws IOException {
            if(pleReader!=null) return new Qwen4PleTable(index,reader,pleReader,prefix,eos);
            long pleBytes=Qwen4PleTable.plePageCacheBytes();
            if(pleBytes==Qwen4PleTable.SHARED_PLE_PAGE_CACHE_BYTES) return new Qwen4PleTable(index,reader,prefix,eos);
            return new Qwen4PleTable(index,reader,pleBytes,prefix,eos);
        }
        public State newState() { return new State(); }
        /** Warm PLE pages for rows a caller already knows are upcoming, without dequantizing. */
        public void prefetchRows(long[] rows) throws IOException { table.prefetchRows(rows); }
        /** Page-cache statistics of the reader paging this layer's PLE rows. */
        public PagedTensorReader.Statistics pleStatistics() { return table.statistics(); }
        @Override public void close() throws IOException { table.close(); }
        public float[] forward(int token,float[] hidden,State state) throws IOException {
            return forward(token,hidden,state,false,false);
        }
        public float[] forward(int token,float[] hidden,State state,boolean fusedDecode) throws IOException {
            return forward(token,hidden,state,fusedDecode,false);
        }
        public float[] apply(int token,float[] hidden,State state,boolean fusedDecode) throws IOException {
            return forward(token,hidden,state,fusedDecode,true);
        }
        /**
         * Distinct-row batched {@link #apply} for prefill: n-gram history pushes and table
         * gathers run in token order, the key/value projections run as one distinct-row
         * GEMM each, and the gating/convolution (or device session) then consumes the rows
         * in the same order, so convolution and history state evolve exactly as sequential
         * {@code apply} calls with {@code fusedDecode == false} would leave them.
         */
        public float[][] applyBatch(int[] tokens,float[][] hiddens,State state) throws IOException {
            state.requireOpen();
            if(state.owner!=this || tokens.length==0 || tokens.length!=hiddens.length) throw new IllegalArgumentException("Invalid batched PLE input/state");
            int rows=tokens.length;
            float[][] embeddings=new float[rows][];
            for(int row=0; row<rows; row++) {
                if(tokens[row]<0 || tokens[row]>=config.vocabularySize() || hiddens[row].length!=config.hyperDim()) throw new IllegalArgumentException("Invalid PLE input/state");
                embeddings[row]=new float[config.pleEmbedDim()];
                table.fetchRows(state.history.push(tokens[row]),embeddings[row]);
            }
            float[][] projectedKeys=unflatten(projectBatch(key,flatten(embeddings,config.pleEmbedDim()),rows),rows,key.rows());
            float[][] values=unflatten(projectBatch(value,flatten(embeddings,config.pleEmbedDim()),rows),rows,value.rows());
            float[][] outputs=new float[rows][];
            for(int row=0; row<rows; row++) outputs[row]=applyProjected(hiddens[row],projectedKeys[row],values[row],state,false,true);
            return outputs;
        }
        private float[] forward(int token,float[] hidden,State state,boolean fusedDecode,boolean addResidual) throws IOException {
            state.requireOpen();
            if(state.owner!=this || token<0 || token>=config.vocabularySize() || hidden.length!=config.hyperDim()) throw new IllegalArgumentException("Invalid PLE input/state");
            float[] embeddings=new float[config.pleEmbedDim()];
            table.fetchRows(state.history.push(token),embeddings);
            float[] projectedKey=project(key,embeddings),values=project(value,embeddings);
            return applyProjected(hidden,projectedKey,values,state,fusedDecode,addResidual);
        }
        private float[] applyProjected(float[] hidden,float[] projectedKey,float[] values,State state,boolean fusedDecode,boolean addResidual) throws IOException {
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

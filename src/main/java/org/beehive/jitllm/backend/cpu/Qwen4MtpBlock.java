package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.*;
import java.util.function.BiConsumer;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.inference.op.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;

/** Shared NextN block. Target HC hidden at t is fused with the embedding of token t+1.
 * Request-owned QSA state supports teacher extension, three frozen-index draft steps and rollback.
 * Embeddings, head and file reader are borrowed from the target model.
 */
public final class Qwen4MtpBlock implements AutoCloseable {
    private final Qwen4Configuration config;
    private final ModelOptMatrix embedding,lmHead;
    private final int validVocabulary,checkpointCapacity;
    private Qwen4CpuLayers.HyperConnection attentionMix,mlpMix,mixer;
    private Qwen4QsaAttention attention;
    private Qwen4Moe moe;
    private MtpFusionExecutor.Fusion fusion;
    private final Set<State> states=Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;
    public Qwen4MtpBlock(Qwen4Configuration trunk,SafeTensorIndex index,PagedTensorReader reader,ModelOptMatrix embedding,ModelOptMatrix lmHead,
                         int validVocabulary,int checkpointCapacity,MatrixExecutor matrices,FlashNextLayerExecutor layers,QsaExecutor qsa,
                         MoeExecutor experts,MtpFusionExecutor fusionExecutor) throws IOException {
        config=trunk.mtpConfiguration();this.embedding=embedding;this.lmHead=lmHead;this.validVocabulary=validVocabulary;this.checkpointCapacity=checkpointCapacity;
        if(validVocabulary<=0 || validVocabulary>config.vocabularySize() || checkpointCapacity<2
                || embedding.rows()!=config.vocabularySize() || embedding.columns()!=config.dim() || lmHead.rows()!=embedding.rows() || lmHead.columns()!=config.dim())
            throw new IllegalArgumentException("Invalid borrowed MTP embedding/head or checkpoints");
        try {
            // HC decode mix/injection deliberately retain F32 projection accumulators.
            // MTP attention's ordinary dense projections store BF16 GEMM results.
            MatrixExecutor projection=matrices==null ? null : new MatrixExecutor() {
                public void multiply(ModelOptMatrix matrix,float[] input,float[] output) throws IOException {
                    if(matrix.weights().name().contains("hyper_connection")) matrices.multiply(matrix,input,output);
                    else matrices.multiplyBf16(matrix,input,output,1);
                }
                public void multiplyBf16(ModelOptMatrix matrix,float[] input,float[] output,int rows) throws IOException {matrices.multiplyBf16(matrix,input,output,rows);}
                public void multiplyBf16Batch(ModelOptMatrix matrix,float[] input,float[] output,int rows) throws IOException {matrices.multiplyBf16Batch(matrix,input,output,rows);}
                public void multiplyNvfp4(ModelOptMatrix matrix,Nvfp4Activation.Packed input,float[] output) throws IOException {matrices.multiplyNvfp4(matrix,input,output);}
            };
            var fcE=Qwen4CpuLayers.matrix(index,reader,"mtp.fc_embedding",config.dim(),config.dim(),matrices);
            var fcH=Qwen4CpuLayers.matrix(index,reader,"mtp.fc_hidden",config.dim(),config.dim(),matrices);
            var en=Qwen4CpuLayers.vector(index,reader,"mtp.pre_fc_norm_embedding.weight",config.dim());
            var hn=Qwen4CpuLayers.vector(index,reader,"mtp.pre_fc_norm_hidden.weight",config.hyperDim());
            fusion=fusionExecutor==null ? hostFusion(fcE,fcH,en,hn) : fusionExecutor.newFusion(fcE,fcH,config.hcCount(),config.rmsNormEps(),en,hn);
            attentionMix=new Qwen4CpuLayers.HyperConnection(config,index,reader,"mtp.layers.0.attn_hyper_connection",true,projection,layers);
            mlpMix=new Qwen4CpuLayers.HyperConnection(config,index,reader,"mtp.layers.0.mlp_hyper_connection",true,projection,layers);
            mixer=new Qwen4CpuLayers.HyperConnection(config,index,reader,"mtp.hyper_connection_mixer",false,projection,layers);
            attention=new Qwen4QsaAttention(config,index,reader,"mtp.layers.0.self_attn",true,true,projection,qsa,3);
            if(!index.require("mtp.layers.0.mlp.experts.0.gate_proj.weight").dtype().equals("F8_E4M3")) throw new IOException("MTP requires block-FP8 experts");
            moe=new Qwen4Moe(config,index,reader,"mtp.layers.0.mlp",matrices,experts);
        } catch(IOException|RuntimeException|Error failure) {try {close();} catch(IOException closing) {failure.addSuppressed(closing);}throw failure;}
    }
    private MtpFusionExecutor.Fusion hostFusion(ModelOptMatrix fcE,ModelOptMatrix fcH,float[] en,float[] hn) {
        return new MtpFusionExecutor.Fusion() {
            boolean released;
            public float[] forward(float[] embeddings,float[] hidden,int rows) throws IOException {
                if(released) throw new IOException("MTP fusion is closed");
                if(rows<=0 || embeddings.length!=rows*config.dim() || hidden.length!=rows*config.hyperDim()) throw new IllegalArgumentException("Invalid MTP fusion batch");
                float[] e=norm(embeddings,en),h=norm(hidden,hn),ep=new float[e.length],hp=new float[h.length];
                fcE.multiplyBf16Batch(e,ep,rows);fcH.multiplyBf16Batch(h,hp,rows*config.hcCount());
                for(int i=0;i<hp.length;i++) hp[i]=FloatFormats.roundBf16(hp[i]+ep[i/config.hyperDim()*config.dim()+i%config.dim()]);return hp;
            }
            public void close() {released=true;}
        };
    }
    private float[] norm(float[] input,float[] weights) {
        int width=weights.length;float[] output=new float[input.length];
        for(int start=0;start<input.length;start+=width) {
            double sum=0;for(int j=0;j<width;j++) {float x=input[start+j];sum+=x*x;}
            float inverse=(float)(1/Math.sqrt((float)(sum/width)+config.rmsNormEps()));
            for(int j=0;j<width;j++) output[start+j]=FloatFormats.roundBf16(input[start+j]*inverse*(1+weights[j]));
        }return output;
    }
    public Qwen4Configuration configuration() {return config;}
    public final class State implements org.beehive.jitllm.inference.state.Qwen4SequenceState {
        private final Qwen4MtpBlock owner=Qwen4MtpBlock.this;
        private final Qwen4QsaAttention.State qsa=attention.newState();
        private final Set<Snapshot> snapshots=Collections.newSetFromMap(new IdentityHashMap<>());
        private int position,prefillStart,prefillEnd,freshEnd,capturedLength;
        private int[] capturedSelection=new int[0];
        private long epoch;
        private boolean drafting,released;
        private void check() {if(released || closed) throw new IllegalStateException("MTP state is closed");}
        public int position() {return position;}
        public int[] selectedTokens() {synchronized(Qwen4MtpBlock.this) {check();return qsa.selectedTokens();}}
        public int liveSnapshots() {synchronized(Qwen4MtpBlock.this) {return snapshots.size();}}
        public void beginPrefill(int tokens) {
            synchronized(Qwen4MtpBlock.this) {check();if(drafting || tokens<=0 || (long)position+tokens>config.contextLength()) throw new IllegalArgumentException("Invalid MTP teacher extend");
                prefillStart=position;prefillEnd=position+tokens;freshEnd=position==0 ? Math.min(4096,prefillEnd) : 0;}
        }
        public void finishPrefill() {synchronized(Qwen4MtpBlock.this) {check();prefillEnd=freshEnd=0;qsa.finishFreshPrefill();}}
        private int rows() {return position>=prefillEnd ? 1 : Math.min(4096,prefillEnd-(position-(position-prefillStart)%4096));}
        public final class Snapshot implements AutoCloseable {
            private final State owner=State.this;
            private final long generation=epoch;
            private final int length=position,start=prefillStart,end=prefillEnd,fresh=freshEnd,capture=capturedLength;
            private final int[] selection=capturedSelection.clone();
            private final boolean draft=drafting;
            private final Qwen4QsaAttention.State.Snapshot attention;
            private boolean disposed,valid=true;
            private Snapshot() throws IOException {attention=qsa.snapshot();}
            public int position() {return length;}
            @Override public void close() {synchronized(Qwen4MtpBlock.this) {if(!disposed) {disposed=true;snapshots.remove(this);attention.close();}}}
        }
        public Snapshot snapshot() throws IOException {
            synchronized(Qwen4MtpBlock.this) {check();if(snapshots.size()>=checkpointCapacity) throw new IllegalStateException("MTP checkpoint capacity exhausted");var saved=new Snapshot();snapshots.add(saved);return saved;}
        }
        public void restore(Snapshot saved) throws IOException {
            synchronized(Qwen4MtpBlock.this) {
                check();if(saved.owner!=this || saved.disposed || !saved.valid || saved.generation!=epoch || saved.length>position) throw new IllegalArgumentException("Invalid MTP checkpoint");
                qsa.restore(saved.attention);for(var future:snapshots) if(future.length>saved.length) future.valid=false;
                position=saved.length;prefillStart=saved.start;prefillEnd=saved.end;freshEnd=saved.fresh;
                capturedLength=saved.capture;capturedSelection=saved.selection.clone();drafting=saved.draft;
            }
        }
        public void reset() {
            synchronized(Qwen4MtpBlock.this) {check();for(var saved:List.copyOf(snapshots)) saved.close();epoch++;qsa.reset();position=prefillStart=prefillEnd=freshEnd=capturedLength=0;capturedSelection=new int[0];drafting=false;}
        }
        @Override public void close() throws IOException {
            synchronized(Qwen4MtpBlock.this) {if(released) return;released=true;for(var saved:List.copyOf(snapshots)) saved.close();qsa.close();states.remove(this);}
        }
    }
    public synchronized State newState() {if(closed) throw new IllegalStateException("MTP block is closed");var state=new State();states.add(state);return state;}
    public synchronized Qwen4CpuEngine.Output forward(int successorToken,float[] targetHidden,State state,boolean logits,BiConsumer<String,float[]> trace) throws IOException {
        return forward(successorToken,targetHidden,state,logits,trace,false);
    }
    public synchronized Qwen4CpuEngine.Output forwardDraft(int successorToken,float[] draftHidden,State state,boolean logits,BiConsumer<String,float[]> trace) throws IOException {
        return forward(successorToken,draftHidden,state,logits,trace,true);
    }
    private Qwen4CpuEngine.Output forward(int token,float[] hidden,State state,boolean withLogits,BiConsumer<String,float[]> trace,boolean draft) throws IOException {
        state.check();if(state.owner!=this || token<0 || token>=validVocabulary || hidden.length!=config.hyperDim() || state.position>=config.contextLength()) throw new IllegalArgumentException("Invalid MTP input/state");
        if(!draft && state.drafting) throw new IllegalArgumentException("Restore teacher prefix before MTP extend");
        if(draft && (state.capturedLength<=0 || state.position-state.capturedLength>=3 || state.prefillEnd>state.position)) throw new IllegalArgumentException("MTP draft requires a captured teacher prefix and at most three steps");
        try(var before=state.snapshot()) {
            try {
                int rows=draft ? 1 : state.rows();float[] row=new float[config.dim()];embedding.readRow(token,row);hidden=fusion.forward(row,hidden,1);emit(trace,"fusion",hidden);
                var mixed=attentionMix.mix(hidden,rows);emit(trace,"attention_mix",mixed.output());
                float[] attended;
                if(draft) {
                    int tail=state.position-state.capturedLength+1;int[] selected=Arrays.copyOf(state.capturedSelection,state.capturedSelection.length+tail);
                    for(int i=0;i<tail;i++) selected[state.capturedSelection.length+i]=state.capturedLength+i;
                    attended=attention.forwardFrozen(mixed.output(),state.qsa,selected);
                } else {if(state.position<state.prefillEnd) state.qsa.setPrefillQueryCount(rows);attended=attention.forward(mixed.output(),state.qsa,state.position<state.prefillEnd,state.position<state.freshEnd);}
                emit(trace,"attention",attended);hidden=attentionMix.combine(attended,mixed);emit(trace,"after_attention",hidden);
                mixed=mlpMix.mix(hidden,rows);emit(trace,"mlp_mix",mixed.output());var routed=moe.forward(mixed.output()).output();emit(trace,"moe",routed);
                hidden=mlpMix.combine(routed,mixed);emit(trace,"hyper_hidden",hidden);float[] output=mixer.mix(hidden,rows).output();emit(trace,"hidden",output);
                float[] logits=withLogits ? Qwen4CpuLayers.project(lmHead,output) : null;
                if(logits!=null) Arrays.fill(logits,validVocabulary,logits.length,Float.NEGATIVE_INFINITY);
                state.position++;state.drafting=draft;
                if(!draft) {state.capturedLength=state.position;state.capturedSelection=state.qsa.selectedTokens();}
                if(state.position==state.freshEnd) state.qsa.finishFreshPrefill();
                return new Qwen4CpuEngine.Output(output,hidden,logits);
            } catch(IOException|RuntimeException|Error failure) {try {state.restore(before);} catch(IOException|RuntimeException|Error rollback) {failure.addSuppressed(rollback);}throw failure;}
        }
    }
    private static void emit(BiConsumer<String,float[]> trace,String name,float[] values) {if(trace!=null) trace.accept(name,values.clone());}
    @Override public synchronized void close() throws IOException {
        if(closed) return;IOException failure=null;
        for(var state:List.copyOf(states)) try {state.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        for(AutoCloseable resource:new AutoCloseable[]{moe,attentionMix,mlpMix,mixer,fusion}) if(resource!=null) try {resource.close();} catch(Exception error) {
            if(failure==null) failure=new IOException("Cannot close MTP block",error);else failure.addSuppressed(error);
        }
        closed=true;if(failure!=null) throw failure;
    }
}

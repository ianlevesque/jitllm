package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.BiConsumer;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;
import org.beehive.jitllm.inference.op.DeltaNetExecutor;

/** Full Flash-Next host forward pass. Weight storage and PLE stay in the original checkpoint. */
public final class Qwen4CpuEngine implements AutoCloseable {
    private record Layer(Qwen4CpuLayers.HyperConnection attentionMix,Qwen4CpuLayers.HyperConnection mlpMix,
                         Qwen4DeltaNet delta,Qwen4QsaAttention attention,Qwen4Moe moe,Qwen4CpuLayers.Ple ple) {}
    public record Output(float[] hidden,float[] hyperHidden,float[] logits) {
        public int greedy() {
            if(logits==null) throw new IllegalStateException("Logits were not requested");
            int best=0;for(int i=1;i<logits.length;i++) if(logits[i]>logits[best]) best=i;
            return best;
        }
    }
    public final class Sequence implements org.beehive.jitllm.inference.state.Qwen4SequenceState {
        private final Qwen4CpuEngine owner=Qwen4CpuEngine.this;
        private final Qwen4DeltaNet.State[] delta=new Qwen4DeltaNet.State[layers.length];
        private final Qwen4QsaAttention.State[] attention=new Qwen4QsaAttention.State[layers.length];
        private final Qwen4CpuLayers.Ple.State[] ple=new Qwen4CpuLayers.Ple.State[layers.length];
        private int position;
        private int freshPrefillEnd;
        private int prefillEnd;
        private int prefillStart;
        private boolean closed;
        private Sequence() {
            try {
                for(int i=0;i<layers.length;i++) {
                    if(layers[i].delta()!=null) delta[i]=layers[i].delta().newState();
                    if(layers[i].attention()!=null) attention[i]=layers[i].attention().newState();
                    if(layers[i].ple()!=null) ple[i]=layers[i].ple().newState();
                }
            } catch(RuntimeException|Error failure) {try {close();} catch(IOException closing) {failure.addSuppressed(closing);}throw failure;}
        }
        private void requireOpen() {if(closed || Qwen4CpuEngine.this.closed) throw new IllegalStateException("Flash-Next sequence is closed");}
        public int position() { return position; }
        public boolean requiresLogits() { return prefillEnd==0 || position+1>=prefillEnd; }
        /** Match SGLang's no-prefix first chunk; later chunks attend through the FP8 pool. */
        public void beginInitialPrefill(int tokens) {
            if(position!=0 || tokens<=0 || tokens>config.contextLength()) throw new IllegalArgumentException("Invalid initial prefill");
            beginPrefill(tokens);
        }
        public void beginPrefill(int tokens) {
            requireOpen();
            if(tokens<=0 || (long)position+tokens>config.contextLength()) throw new IllegalArgumentException("Invalid prefill length");
            prefillEnd=position+tokens;
            prefillStart=position;
            freshPrefillEnd=position==0 ? Math.min(tokens,4096) : 0;
        }
        public void finishPrefill() {
            prefillEnd=0;freshPrefillEnd=0;
            for(var state:attention) if(state!=null) state.finishFreshPrefill();
        }
        public record Snapshot(Sequence owner,int position,int freshPrefillEnd,int prefillEnd,int prefillStart,Qwen4DeltaNet.State.Snapshot[] delta,Qwen4QsaAttention.State.Snapshot[] attention,Qwen4CpuLayers.Ple.State.Snapshot[] ple) implements AutoCloseable {
            @Override public void close() {for(var snapshot:delta) if(snapshot!=null) snapshot.close();}
        }
        public Snapshot snapshot() throws IOException {
            requireOpen();
            var ds=new Qwen4DeltaNet.State.Snapshot[layers.length];var as=new Qwen4QsaAttention.State.Snapshot[layers.length];var ps=new Qwen4CpuLayers.Ple.State.Snapshot[layers.length];
            try {
                for(int i=0;i<layers.length;i++) {if(delta[i]!=null) ds[i]=delta[i].snapshot();if(attention[i]!=null) as[i]=attention[i].snapshot();if(ple[i]!=null) ps[i]=ple[i].snapshot();}
            } catch(IOException|RuntimeException|Error failure) {for(var snapshot:ds) if(snapshot!=null) snapshot.close();throw failure;}
            return new Snapshot(this,position,freshPrefillEnd,prefillEnd,prefillStart,ds,as,ps);
        }
        public void restore(Snapshot snapshot) throws IOException {
            requireOpen();
            if(snapshot.owner()!=this || snapshot.delta().length!=layers.length || snapshot.attention().length!=layers.length || snapshot.ple().length!=layers.length)
                throw new IllegalArgumentException("Wrong model state snapshot");
            for(int i=0;i<layers.length;i++) {if(delta[i]!=null) delta[i].restore(snapshot.delta()[i]);if(attention[i]!=null) attention[i].restore(snapshot.attention()[i]);if(ple[i]!=null) ple[i].restore(snapshot.ple()[i]);}
            position=snapshot.position();
            freshPrefillEnd=snapshot.freshPrefillEnd();
            prefillEnd=snapshot.prefillEnd();
            prefillStart=snapshot.prefillStart();
        }
        public void reset() {
            requireOpen();
            for(int i=0;i<layers.length;i++) {if(delta[i]!=null) delta[i].reset();if(attention[i]!=null) attention[i].reset();if(ple[i]!=null) ple[i].reset();}
            position=0;freshPrefillEnd=0;prefillEnd=0;prefillStart=0;
        }
        @Override public void close() throws IOException {
            if(closed) return;closed=true;IOException failure=null;
            for(var state:delta) if(state!=null) try {state.close();} catch(IOException closing) {
                if(failure==null) failure=closing;else failure.addSuppressed(closing);
            }
            synchronized(Qwen4CpuEngine.this) {sequences.remove(this);}if(failure!=null) throw failure;
        }
    }
    private final Qwen4Configuration config;
    private final Qwen4Tokenizer tokenizer;
    private final PagedTensorReader reader;
    private final ModelOptMatrix embedding,lmHead;
    private final Qwen4CpuLayers.HyperConnection mixer;
    private final Layer[] layers;
    private final Set<Sequence> sequences=Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded) throws IOException {
        this(directory,context,pageBudget,onLoaded,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor) throws IOException {
        this(directory,context,pageBudget,onLoaded,executor,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor,DeltaNetExecutor recurrentExecutor) throws IOException {
        config=Qwen4Configuration.load(directory,context);
        tokenizer=Qwen4Tokenizer.load(directory);
        reader=new PagedTensorReader(pageBudget,4096);
        try {
            var index=SafeTensorIndex.open(directory,name->name.startsWith("model.language_model.") || name.equals("lm_head.weight"));
            String root="model.language_model";
            embedding=Qwen4CpuLayers.matrix(index,reader,root+".embed_tokens",config.vocabularySize(),config.dim(),executor);
            lmHead=Qwen4CpuLayers.matrix(index,reader,"lm_head",config.vocabularySize(),config.dim(),executor);
            mixer=new Qwen4CpuLayers.HyperConnection(config,index,reader,root+".hyper_connection_mixer",false,executor);
            layers=new Layer[config.numberOfLayers()];
            for(int i=0;i<layers.length;i++) {
                String prefix=root+".layers."+i;
                var attentionMix=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".attn_hyper_connection",true,executor);
                var mlpMix=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".mlp_hyper_connection",true,executor);
                var delta=config.isRecurrentLayer(i) ? new Qwen4DeltaNet(config,index,reader,prefix+".linear_attn",executor,recurrentExecutor) : null;
                var attention=config.isRecurrentLayer(i) ? null : new Qwen4QsaAttention(config,index,reader,prefix+".self_attn",true,true,executor);
                var moe=new Qwen4Moe(config,index,reader,prefix+".mlp",executor);
                var ple=config.hasPle(i) ? new Qwen4CpuLayers.Ple(config,index,reader,prefix+".ple",executor) : null;
                layers[i]=new Layer(attentionMix,mlpMix,delta,attention,moe,ple);
                if(onLoaded!=null) onLoaded.accept(i+1,layers.length);
            }
        } catch(IOException|RuntimeException|Error failure) {
            try {reader.close();} catch(IOException closing) {failure.addSuppressed(closing);} throw failure;
        }
    }
    public Qwen4Configuration configuration() { return config; }
    public Qwen4Tokenizer tokenizer() { return tokenizer; }
    public synchronized Sequence newSequence() { if(closed) throw new IllegalStateException("Engine is closed");var sequence=new Sequence();sequences.add(sequence);return sequence; }
    public PagedTensorReader.Statistics statistics() { return reader.statistics(); }
    public Output forward(int token,Sequence sequence,boolean withLogits,BiConsumer<Integer,float[]> onLayer) throws IOException {
        if(closed || sequence.closed || sequence.owner!=this || token<0 || token>=tokenizer.tokenCount() || sequence.position>=config.contextLength())
            throw new IllegalArgumentException("Invalid model input/state/context");
        var before=sequence.snapshot();
        try {
            float[] row=new float[config.dim()];embedding.readRow(token,row);
            float[] hidden=new float[config.hyperDim()];
            for(int branch=0;branch<config.hcCount();branch++) System.arraycopy(row,0,hidden,branch*row.length,row.length);
            for(int i=0;i<layers.length;i++) {
                var layer=layers[i];
                if(sequence.attention[i]!=null && sequence.position<sequence.prefillEnd) {
                    int chunkStart=sequence.position-(sequence.position-sequence.prefillStart)%4096;
                    sequence.attention[i].setPrefillQueryCount(Math.min(4096,sequence.prefillEnd-chunkStart));
                }
                if(layer.ple()!=null) {
                    float[] added=layer.ple().forward(token,hidden,sequence.ple[i]);
                    for(int j=0;j<hidden.length;j++) hidden[j]=FloatFormats.roundBf16(hidden[j]+added[j]);
                }
                var mixed=layer.attentionMix().mix(hidden);
                float[] attention=layer.delta()!=null ? layer.delta().forward(mixed.output(),sequence.delta[i])
                        : layer.attention().forward(mixed.output(),sequence.attention[i],sequence.position<sequence.prefillEnd,sequence.position<sequence.freshPrefillEnd);
                hidden=layer.attentionMix().combine(attention,mixed);
                mixed=layer.mlpMix().mix(hidden);
                hidden=layer.mlpMix().combine(layer.moe().forward(mixed.output()).output(),mixed);
                if(onLayer!=null) onLayer.accept(i,hidden.clone());
            }
            float[] output=mixer.mix(hidden).output(),logits=null;
            if(withLogits) {
                logits=Qwen4CpuLayers.project(lmHead,output);
                Arrays.fill(logits,tokenizer.tokenCount(),logits.length,Float.NEGATIVE_INFINITY);
            }
            sequence.position++;
            if(sequence.position==sequence.freshPrefillEnd)
                for(var attention:sequence.attention) if(attention!=null) attention.finishFreshPrefill();
            return new Output(output,hidden,logits);
        } catch(IOException|RuntimeException|Error failure) {
            try {sequence.restore(before);} catch(IOException|RuntimeException|Error rollback) {failure.addSuppressed(rollback);}throw failure;
        } finally {before.close();}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var sequence:List.copyOf(sequences)) try {sequence.close();} catch(IOException closing) {
            if(failure==null) failure=closing;else failure.addSuppressed(closing);
        }
        try {reader.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        if(failure!=null) throw failure;
    }
}

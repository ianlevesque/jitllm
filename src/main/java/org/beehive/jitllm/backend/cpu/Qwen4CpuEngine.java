package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.BiConsumer;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.loader.Qwen4PleTable;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;
import org.beehive.jitllm.inference.op.DeltaNetExecutor;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor;
import org.beehive.jitllm.inference.op.QsaExecutor;
import org.beehive.jitllm.inference.op.MoeExecutor;
import org.beehive.jitllm.inference.state.MultimodalPromptPlan;
import org.beehive.jitllm.model.vision.FlashNextImagePreprocessor;
import org.beehive.jitllm.model.vision.FlashNextVisionConfig;
import org.beehive.jitllm.model.vision.FlashNextVisionTower;
import org.beehive.jitllm.model.vision.FlashNextVisionWeights;
import org.beehive.jitllm.model.vision.PreprocessedImage;

/** Full Flash-Next host forward pass. Weight storage and PLE stay in the original checkpoint. */
public final class Qwen4CpuEngine implements AutoCloseable {
    /**
     * Pure prefill geometry planning. Today every token in a 4096-token chunk reports the
     * whole remaining chunk size as its GEMM geometry (matching staged chunked prefill with
     * chunk 4096). {@code jitllm.qwen4.prefillChunk} &gt; 1 caps that geometry at the given
     * chunk size, aligned to the prefill start; values &lt;= 1 keep the legacy 4096 geometry
     * bit-identically. Only the reported row geometry changes: tokens are still forwarded
     * one at a time, so DeltaNet recurrence, QSA appends and PLE state see the same order.
     */
    public static final class PrefillGeometry {
        public static final int LEGACY_CHUNK=4096;
        private PrefillGeometry() {}
        public static int configuredChunk() {
            int chunk=Integer.getInteger("jitllm.qwen4.prefillChunk",1);
            if(chunk<1) throw new IllegalArgumentException("jitllm.qwen4.prefillChunk must be at least 1");
            return chunk;
        }
        /** Geometry chunk actually used; {@code configured <= 1} selects the legacy geometry. */
        public static int effectiveChunk(int configured) {
            if(configured<1) throw new IllegalArgumentException("Prefill chunk must be at least 1");
            return configured<=1 ? LEGACY_CHUNK : Math.min(configured,LEGACY_CHUNK);
        }
        public static int chunkStart(int position,int prefillStart,int configured) {
            return position-(position-prefillStart)%effectiveChunk(configured);
        }
        /** Rows reported for the token at {@code position}; 1 outside prefill (decode). */
        public static int rows(int position,int prefillStart,int prefillEnd,int configured) {
            if(position>=prefillEnd) return 1;
            return Math.min(effectiveChunk(configured),prefillEnd-chunkStart(position,prefillStart,configured));
        }
    }
    /**
     * Distinct-row batched prefill planning, gated by {@code jitllm.qwen4.batchedPrefill}
     * (default off: every caller keeps the sequential per-token path operation for
     * operation). When enabled, {@code jitllm.qwen4.prefillRows} (default 32) caps how
     * many prompt rows one chunk forwards layer by layer. A chunk never crosses the
     * prefill end, the fresh-prefill boundary, or a {@link PrefillGeometry} chunk
     * boundary, and geometry of 16 rows or fewer never batches: that geometry selects the
     * fused FP32-accumulator HC path, whose arithmetic the batched BF16 GEMMs would not
     * reproduce. Within a chunk every stateful op still consumes rows in token order.
     */
    public static final class BatchedPrefill {
        private BatchedPrefill() {}
        public static boolean enabled() { return Boolean.getBoolean("jitllm.qwen4.batchedPrefill"); }
        public static int configuredRows() {
            int rows=Integer.getInteger("jitllm.qwen4.prefillRows",32);
            if(rows<1) throw new IllegalArgumentException("jitllm.qwen4.prefillRows must be at least 1");
            return rows;
        }
        public static int effectiveRows(int configured) {
            if(configured<1) throw new IllegalArgumentException("Prefill rows must be at least 1");
            return Math.min(configured,PrefillGeometry.LEGACY_CHUNK);
        }
        /**
         * Rows the chunk starting at {@code position} may batch, or 0 when the caller must
         * forward sequentially (outside prefill, fused geometry, or a lone final row).
         */
        public static int chunkLength(int position,int prefillStart,int prefillEnd,int freshEnd,int configuredRows,int configuredGeometryChunk) {
            if(position>=prefillEnd) return 0;
            int geometry=PrefillGeometry.rows(position,prefillStart,prefillEnd,configuredGeometryChunk);
            if(geometry<=16) return 0;
            int limit=prefillEnd;
            if(freshEnd>position) limit=Math.min(limit,freshEnd);
            limit=Math.min(limit,PrefillGeometry.chunkStart(position,prefillStart,configuredGeometryChunk)+geometry);
            int length=Math.min(effectiveRows(configuredRows),limit-position);
            return length>=2 ? length : 0;
        }
    }
    public record ForwardStatistics(long forwards,long prefillForwards,long rollbacks) {}
    /** Batched-prefill counters; row counts are per layer invocation, as forwarded. */
    public record BatchedPrefillStatistics(long chunks,long rows,long embeddingRows,long pleRows,long hyperRows,long deltaRows,long qsaRows,long moeRows,long fallbacks) {}
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
        /** Staged image rows/MRoPE for the prompt being ingested, or null on the text path. */
        private MultimodalPromptPlan multimodalPlan;
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
        /**
         * Stages the image plan the next forwards splice in. Only at position zero: the plan
         * is indexed by absolute prompt position, so staging it mid-sequence would misalign
         * every row and MRoPE triple after it.
         */
        @Override public void stageMultimodalPrompt(MultimodalPromptPlan plan) {
            requireOpen();
            java.util.Objects.requireNonNull(plan,"plan");
            if(position!=0) throw new IllegalStateException("Stage an image prompt before ingesting it");
            if(plan.promptLength()<=0 || plan.promptLength()>config.contextLength())
                throw new IllegalArgumentException("Invalid image prompt length");
            multimodalPlan=plan;
        }
        /** The staged image plan, or null on the text path. */
        public MultimodalPromptPlan multimodalPlan() { return multimodalPlan; }
        public boolean requiresLogits() { return prefillEnd==0 || position+1>=prefillEnd; }
        public int prefillRows() {
            return PrefillGeometry.rows(position,prefillStart,prefillEnd,prefillChunk);
        }
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
        public record Snapshot(Sequence owner,int position,int freshPrefillEnd,int prefillEnd,int prefillStart,MultimodalPromptPlan multimodalPlan,Qwen4DeltaNet.State.Snapshot[] delta,Qwen4QsaAttention.State.Snapshot[] attention,Qwen4CpuLayers.Ple.State.Snapshot[] ple) implements AutoCloseable {
            @Override public void close() {
                for(var snapshot:delta) if(snapshot!=null) snapshot.close();
                for(var snapshot:ple) if(snapshot!=null) snapshot.close();
                for(var snapshot:attention) if(snapshot!=null) snapshot.close();
            }
        }
        public Snapshot snapshot() throws IOException {
            requireOpen();
            var ds=new Qwen4DeltaNet.State.Snapshot[layers.length];var as=new Qwen4QsaAttention.State.Snapshot[layers.length];var ps=new Qwen4CpuLayers.Ple.State.Snapshot[layers.length];
            try {
                for(int i=0;i<layers.length;i++) {if(delta[i]!=null) ds[i]=delta[i].snapshot();if(attention[i]!=null) as[i]=attention[i].snapshot();if(ple[i]!=null) ps[i]=ple[i].snapshot();}
            } catch(IOException|RuntimeException|Error failure) {
                for(var snapshot:ds) if(snapshot!=null) snapshot.close();
                for(var snapshot:ps) if(snapshot!=null) snapshot.close();
                for(var snapshot:as) if(snapshot!=null) snapshot.close();throw failure;
            }
            return new Snapshot(this,position,freshPrefillEnd,prefillEnd,prefillStart,multimodalPlan,ds,as,ps);
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
            // The plan is immutable and read by position, so restoring the reference (with
            // the position above) restores exactly the rows/MRoPE the replay will consume.
            multimodalPlan=snapshot.multimodalPlan();
        }
        public void reset() {
            requireOpen();
            for(int i=0;i<layers.length;i++) {if(delta[i]!=null) delta[i].reset();if(attention[i]!=null) attention[i].reset();if(ple[i]!=null) ple[i].reset();}
            position=0;freshPrefillEnd=0;prefillEnd=0;prefillStart=0;multimodalPlan=null;
        }
        @Override public void close() throws IOException {
            if(closed) return;closed=true;IOException failure=null;
            for(var state:delta) if(state!=null) try {state.close();} catch(IOException closing) {
                if(failure==null) failure=closing;else failure.addSuppressed(closing);
            }
            for(var state:ple) if(state!=null) try {state.close();} catch(IOException closing) {
                if(failure==null) failure=closing;else failure.addSuppressed(closing);
            }
            for(var state:attention) if(state!=null) try {state.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
            synchronized(Qwen4CpuEngine.this) {sequences.remove(this);}if(failure!=null) throw failure;
        }
    }
    private final Qwen4Configuration config;
    private final Path checkpointDirectory;
    private final Qwen4Tokenizer tokenizer;
    private final PagedTensorReader reader;
    /** Dedicated PLE page cache shared by every layer's table, or null for the legacy shared reader. */
    private final PagedTensorReader pleReader;
    private final ModelOptMatrix embedding,lmHead;
    private final Qwen4CpuLayers.HyperConnection mixer;
    private final Layer[] layers;
    private final Set<Sequence> sequences=Collections.newSetFromMap(new IdentityHashMap<>());
    private final List<Qwen4CpuLayers.HyperConnection> hyperResources=new ArrayList<>();
    private final List<Qwen4CpuLayers.Ple> pleResources=new ArrayList<>();
    private final List<Qwen4Moe> moeResources=new ArrayList<>();
    private final List<Qwen4MtpBlock> mtpResources=new ArrayList<>();
    private final int prefillChunk=PrefillGeometry.configuredChunk();
    private final boolean batchedPrefill=BatchedPrefill.enabled();
    private final int batchedPrefillRows=BatchedPrefill.configuredRows();
    private final java.util.concurrent.atomic.AtomicLong forwards=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong prefillForwards=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong rollbacks=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedChunks=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedEmbeddingRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedPleRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedHyperRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedDeltaRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedQsaRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedMoeRows=new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong batchedFallbacks=new java.util.concurrent.atomic.AtomicLong();
    private boolean closed;
    /** Lazily loaded vision tower state; a text-only engine never reads the visual weights. */
    private FlashNextVisionConfig visionConfig;
    private FlashNextImagePreprocessor visionPreprocessor;
    private FlashNextVisionTower visionTower;

    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded) throws IOException {
        this(directory,context,pageBudget,onLoaded,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor) throws IOException {
        this(directory,context,pageBudget,onLoaded,executor,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor,DeltaNetExecutor recurrentExecutor) throws IOException {
        this(directory,context,pageBudget,onLoaded,executor,recurrentExecutor,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor,DeltaNetExecutor recurrentExecutor,FlashNextLayerExecutor layerExecutor) throws IOException {
        this(directory,context,pageBudget,onLoaded,executor,recurrentExecutor,layerExecutor,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor,DeltaNetExecutor recurrentExecutor,FlashNextLayerExecutor layerExecutor,QsaExecutor qsaExecutor) throws IOException {
        this(directory,context,pageBudget,onLoaded,executor,recurrentExecutor,layerExecutor,qsaExecutor,null);
    }
    public Qwen4CpuEngine(Path directory,int context,long pageBudget,BiConsumer<Integer,Integer> onLoaded,MatrixExecutor executor,DeltaNetExecutor recurrentExecutor,FlashNextLayerExecutor layerExecutor,QsaExecutor qsaExecutor,MoeExecutor moeExecutor) throws IOException {
        checkpointDirectory=directory.toAbsolutePath().normalize();
        config=Qwen4Configuration.load(directory,context);
        tokenizer=Qwen4Tokenizer.load(directory);
        reader=new PagedTensorReader(pageBudget,4096);
        long pleBytes=Qwen4PleTable.plePageCacheBytes();
        pleReader=pleBytes==Qwen4PleTable.SHARED_PLE_PAGE_CACHE_BYTES ? null : new PagedTensorReader(pleBytes,4096);
        try {
            var index=SafeTensorIndex.open(directory,name->name.startsWith("model.language_model.") || name.equals("lm_head.weight"));
            String root="model.language_model";
            embedding=Qwen4CpuLayers.matrix(index,reader,root+".embed_tokens",config.vocabularySize(),config.dim(),executor);
            lmHead=Qwen4CpuLayers.matrix(index,reader,"lm_head",config.vocabularySize(),config.dim(),executor);
            mixer=own(new Qwen4CpuLayers.HyperConnection(config,index,reader,root+".hyper_connection_mixer",false,executor,layerExecutor));
            layers=new Layer[config.numberOfLayers()];
            for(int i=0;i<layers.length;i++) {
                String prefix=root+".layers."+i;
                var attentionMix=own(new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".attn_hyper_connection",true,executor,layerExecutor));
                var mlpMix=own(new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".mlp_hyper_connection",true,executor,layerExecutor));
                var delta=config.isRecurrentLayer(i) ? new Qwen4DeltaNet(config,index,reader,prefix+".linear_attn",executor,recurrentExecutor) : null;
                var attention=config.isRecurrentLayer(i) ? null : new Qwen4QsaAttention(config,index,reader,prefix+".self_attn",true,true,executor,qsaExecutor);
                var moe=own(new Qwen4Moe(config,index,reader,prefix+".mlp",executor,moeExecutor));
                var ple=config.hasPle(i) ? own(new Qwen4CpuLayers.Ple(config,index,reader,pleReader,prefix+".ple",executor,layerExecutor)) : null;
                layers[i]=new Layer(attentionMix,mlpMix,delta,attention,moe,ple);
                if(onLoaded!=null) onLoaded.accept(i+1,layers.length);
            }
        } catch(IOException|RuntimeException|Error failure) {
            for(var resource:hyperResources) try {resource.close();} catch(IOException closing) {failure.addSuppressed(closing);}
            for(var resource:moeResources) try {resource.close();} catch(IOException closing) {failure.addSuppressed(closing);}
            for(var resource:pleResources) try {resource.close();} catch(IOException closing) {failure.addSuppressed(closing);}
            if(pleReader!=null) try {pleReader.close();} catch(IOException closing) {failure.addSuppressed(closing);}
            try {reader.close();} catch(IOException closing) {failure.addSuppressed(closing);} throw failure;
        }
    }
    private Qwen4CpuLayers.Ple own(Qwen4CpuLayers.Ple resource) {pleResources.add(resource);return resource;}
    private Qwen4Moe own(Qwen4Moe resource) {moeResources.add(resource);return resource;}
    private Qwen4CpuLayers.HyperConnection own(Qwen4CpuLayers.HyperConnection resource) {hyperResources.add(resource);return resource;}
    public Qwen4Configuration configuration() { return config; }
    public synchronized boolean isClosed() { return closed; }
    public Qwen4Tokenizer tokenizer() { return tokenizer; }
    /** The checkpoint's vision configuration, loaded on first image use. */
    public synchronized FlashNextVisionConfig visionConfig() throws IOException {
        if(closed) throw new IllegalStateException("Engine is closed");
        if(visionConfig==null) {
            visionConfig=FlashNextVisionConfig.load(checkpointDirectory);
            if(visionConfig.outHiddenSize()!=config.dim())
                throw new IOException("Vision output width "+visionConfig.outHiddenSize()
                        +" does not match language-model width "+config.dim());
            visionPreprocessor=new FlashNextImagePreprocessor(visionConfig);
        }
        return visionConfig;
    }
    /** Decode and patchify one encoded image exactly as the tower's processor does. */
    public PreprocessedImage preprocessImage(byte[] encodedBytes) throws IOException {
        visionConfig();
        synchronized(this) {return visionPreprocessor.preprocess(encodedBytes);}
    }
    /** Tower rows for one preprocessed image: {@code [mergedTokens][dim]}. Weights load once. */
    public float[][] imageEmbeddings(PreprocessedImage image) throws IOException {
        synchronized(this) {
            visionConfig();
            if(visionTower==null) {
                var index=SafeTensorIndex.open(checkpointDirectory,name->name.startsWith("model.visual."));
                visionTower=new FlashNextVisionTower(FlashNextVisionWeights.load(visionConfig,index,reader));
            }
            return visionTower.forward(image);
        }
    }
    public synchronized Sequence newSequence() { if(closed) throw new IllegalStateException("Engine is closed");var sequence=new Sequence();sequences.add(sequence);return sequence; }
    public PagedTensorReader.Statistics statistics() { return reader.statistics(); }
    /** Configured {@code jitllm.qwen4.prefillChunk}; 1 keeps the legacy 4096 geometry. */
    public int prefillChunk() { return prefillChunk; }
    public ForwardStatistics forwardStatistics() { return new ForwardStatistics(forwards.get(),prefillForwards.get(),rollbacks.get()); }
    /** Whether {@code jitllm.qwen4.batchedPrefill} was set when this engine was built. */
    public boolean batchedPrefillEnabled() { return batchedPrefill; }
    /** Configured {@code jitllm.qwen4.prefillRows} chunk cap for batched prefill. */
    public int batchedPrefillRows() { return batchedPrefillRows; }
    public BatchedPrefillStatistics batchedPrefillStatistics() {
        return new BatchedPrefillStatistics(batchedChunks.get(),batchedRows.get(),batchedEmbeddingRows.get(),batchedPleRows.get(),
                batchedHyperRows.get(),batchedDeltaRows.get(),batchedQsaRows.get(),batchedMoeRows.get(),batchedFallbacks.get());
    }
    public synchronized Qwen4MtpBlock newMtpBlock(int checkpointCapacity,MatrixExecutor matrices,FlashNextLayerExecutor layers,QsaExecutor qsa,MoeExecutor experts,org.beehive.jitllm.inference.op.MtpFusionExecutor fusion) throws IOException {
        if(closed) throw new IllegalStateException("Engine is closed");
        var block=new Qwen4MtpBlock(config,SafeTensorIndex.open(checkpointDirectory,name->name.startsWith("mtp.")),reader,embedding,lmHead,tokenizer.tokenCount(),checkpointCapacity,matrices,layers,qsa,experts,fusion);
        mtpResources.add(block);return block;
    }
    public Output forward(int token,Sequence sequence,boolean withLogits,BiConsumer<Integer,float[]> onLayer) throws IOException {
        // With no staged plan this is the original text path, operation for operation: the
        // overrides below stay null and every branch takes its pre-vision form.
        MultimodalPromptPlan plan=sequence==null ? null : sequence.multimodalPlan;
        float[] embeddingOverride=plan==null ? null : plan.embeddingAt(sequence.position);
        int[] ropeOverride=plan==null ? null : plan.ropeAt(sequence.position);
        return forwardSpliced(token,embeddingOverride,ropeOverride,sequence,withLogits,onLayer);
    }
    /**
     * One forward whose input row and MRoPE triple are supplied instead of derived: the
     * tower row replaces the token-embedding lookup (broadcast across every hyper-connection
     * stream exactly as a looked-up row is), while PLE and the n-gram hash still see {@code
     * token} — the image placeholder id — as the staged runtime presents it. Either override
     * may be null independently; both null is the plain text forward.
     */
    public Output forward(int token,float[] embeddingOverride,int[] ropePositions,Sequence sequence,boolean withLogits,BiConsumer<Integer,float[]> onLayer) throws IOException {
        return forwardSpliced(token,embeddingOverride,ropePositions,sequence,withLogits,onLayer);
    }
    private Output forwardSpliced(int token,float[] embeddingOverride,int[] ropeOverride,Sequence sequence,boolean withLogits,BiConsumer<Integer,float[]> onLayer) throws IOException {
        if(closed || sequence.closed || sequence.owner!=this || token<0 || token>=tokenizer.tokenCount() || sequence.position>=config.contextLength()
                || embeddingOverride!=null && embeddingOverride.length!=config.dim()
                || ropeOverride!=null && ropeOverride.length!=3)
            throw new IllegalArgumentException("Invalid model input/state/context");
        forwards.incrementAndGet();
        if(sequence.position<sequence.prefillEnd) prefillForwards.incrementAndGet();
        var before=sequence.snapshot();
        try {
            int prefillRows=sequence.prefillRows();
            float[] row=new float[config.dim()];
            if(embeddingOverride==null) embedding.readRow(token,row);
            else System.arraycopy(embeddingOverride,0,row,0,row.length);
            float[] hidden=new float[config.hyperDim()];
            for(int branch=0;branch<config.hcCount();branch++) System.arraycopy(row,0,hidden,branch*row.length,row.length);
            for(int i=0;i<layers.length;i++) {
                var layer=layers[i];
                if(sequence.attention[i]!=null && sequence.position<sequence.prefillEnd) {
                    sequence.attention[i].setPrefillQueryCount(prefillRows);
                }
                if(layer.ple()!=null) {
                    hidden=layer.ple().apply(token,hidden,sequence.ple[i],sequence.position>=sequence.prefillEnd);
                }
                var mixed=layer.attentionMix().mix(hidden,prefillRows);
                float[] attention=layer.delta()!=null ? layer.delta().forward(mixed.output(),sequence.delta[i])
                        : ropeOverride==null
                                ? layer.attention().forward(mixed.output(),sequence.attention[i],sequence.position<sequence.prefillEnd,sequence.position<sequence.freshPrefillEnd)
                                : layer.attention().forward(mixed.output(),ropeOverride,sequence.attention[i],sequence.position<sequence.prefillEnd,sequence.position<sequence.freshPrefillEnd);
                hidden=layer.attentionMix().combine(attention,mixed);
                mixed=layer.mlpMix().mix(hidden,prefillRows);
                hidden=layer.mlpMix().combine(layer.moe().forward(mixed.output()).output(),mixed);
                if(onLayer!=null) onLayer.accept(i,hidden.clone());
            }
            float[] output=mixer.mix(hidden,prefillRows).output(),logits=null;
            if(withLogits) {
                logits=Qwen4CpuLayers.project(lmHead,output);
                Arrays.fill(logits,tokenizer.tokenCount(),logits.length,Float.NEGATIVE_INFINITY);
            }
            sequence.position++;
            if(sequence.position==sequence.freshPrefillEnd)
                for(var attention:sequence.attention) if(attention!=null) attention.finishFreshPrefill();
            return new Output(output,hidden,logits);
        } catch(IOException|RuntimeException|Error failure) {
            rollbacks.incrementAndGet();
            try {sequence.restore(before);} catch(IOException|RuntimeException|Error rollback) {failure.addSuppressed(rollback);}throw failure;
        } finally {before.close();}
    }
    /**
     * Forwards {@code tokens} of an open prefill (see {@code Sequence.beginPrefill}),
     * batching distinct rows layer by layer where {@link BatchedPrefill} allows and
     * delegating to sequential {@link #forward} everywhere else. With batched prefill
     * disabled, a staged multimodal plan, or a layer observer (whose token-major callback
     * order batching would change), this is exactly the sequential path. {@code
     * withLogits} requests logits for the final token only, matching how prefill callers
     * drive {@code forward}. A batched chunk snapshots once and rolls the whole chunk back
     * on failure; the state after a successful chunk, including the snapshot a caller
     * takes at {@code finishPrefill}, is the state sequential forwards would have left.
     */
    public Output[] forwardPrefillChunk(int[] tokens,Sequence sequence,boolean withLogits,BiConsumer<Integer,float[]> onLayer) throws IOException {
        if(closed || sequence.closed || sequence.owner!=this || tokens.length==0 || sequence.position+tokens.length>sequence.prefillEnd)
            throw new IllegalArgumentException("Invalid model input/state/context");
        for(int token:tokens) if(token<0 || token>=tokenizer.tokenCount()) throw new IllegalArgumentException("Invalid model input/state/context");
        Output[] outputs=new Output[tokens.length];
        boolean sequential=!batchedPrefill || onLayer!=null || sequence.multimodalPlan!=null;
        if(sequential && batchedPrefill) batchedFallbacks.incrementAndGet();
        int offset=0;
        while(offset<tokens.length) {
            int length=sequential ? 0 : Math.min(tokens.length-offset,
                    BatchedPrefill.chunkLength(sequence.position,sequence.prefillStart,sequence.prefillEnd,sequence.freshPrefillEnd,batchedPrefillRows,prefillChunk));
            if(length<2) {
                outputs[offset]=forward(tokens[offset],sequence,withLogits && offset==tokens.length-1,onLayer);
                offset++;
            } else {
                forwardChunkBatched(tokens,offset,length,sequence,outputs,withLogits && offset+length==tokens.length);
                offset+=length;
            }
        }
        return outputs;
    }
    private void forwardChunkBatched(int[] tokens,int offset,int length,Sequence sequence,Output[] outputs,boolean withLogits) throws IOException {
        int startPosition=sequence.position;
        int geometryRows=sequence.prefillRows();
        boolean fresh=startPosition<sequence.freshPrefillEnd;
        forwards.addAndGet(length);
        prefillForwards.addAndGet(length);
        var before=sequence.snapshot();
        try {
            int[] chunkTokens=Arrays.copyOfRange(tokens,offset,offset+length);
            float[][] hidden=new float[length][];
            for(int row=0; row<length; row++) {
                float[] embeddingRow=new float[config.dim()];
                embedding.readRow(chunkTokens[row],embeddingRow);
                hidden[row]=new float[config.hyperDim()];
                for(int branch=0; branch<config.hcCount(); branch++) System.arraycopy(embeddingRow,0,hidden[row],branch*embeddingRow.length,embeddingRow.length);
            }
            batchedEmbeddingRows.addAndGet(length);
            for(int i=0; i<layers.length; i++) {
                var layer=layers[i];
                if(sequence.attention[i]!=null) sequence.attention[i].setPrefillQueryCount(geometryRows);
                if(layer.ple()!=null) {
                    hidden=layer.ple().applyBatch(chunkTokens,hidden,sequence.ple[i]);
                    batchedPleRows.addAndGet(length);
                }
                var mixed=layer.attentionMix().mixBatch(hidden,geometryRows);
                batchedHyperRows.addAndGet(length);
                float[][] mixedOutputs=new float[length][];
                for(int row=0; row<length; row++) mixedOutputs[row]=mixed[row].output();
                float[][] attention;
                if(layer.delta()!=null) {
                    attention=layer.delta().forwardBatch(mixedOutputs,sequence.delta[i]);
                    batchedDeltaRows.addAndGet(length);
                } else {
                    int[][] ropes=new int[length][];
                    for(int row=0; row<length; row++) ropes[row]=new int[]{startPosition+row,startPosition+row,startPosition+row};
                    attention=layer.attention().forwardBatch(mixedOutputs,ropes,sequence.attention[i],true,fresh);
                    batchedQsaRows.addAndGet(length);
                }
                hidden=layer.attentionMix().combineBatch(attention,mixed);
                mixed=layer.mlpMix().mixBatch(hidden,geometryRows);
                batchedHyperRows.addAndGet(length);
                for(int row=0; row<length; row++) mixedOutputs[row]=mixed[row].output();
                var moeResults=layer.moe().forwardBatch(mixedOutputs);
                batchedMoeRows.addAndGet(length);
                float[][] moeOutputs=new float[length][];
                for(int row=0; row<length; row++) moeOutputs[row]=moeResults[row].output();
                hidden=layer.mlpMix().combineBatch(moeOutputs,mixed);
            }
            var mixed=mixer.mixBatch(hidden,geometryRows);
            batchedHyperRows.addAndGet(length);
            for(int row=0; row<length; row++) {
                float[] logits=null;
                if(withLogits && row==length-1) {
                    logits=Qwen4CpuLayers.project(lmHead,mixed[row].output());
                    Arrays.fill(logits,tokenizer.tokenCount(),logits.length,Float.NEGATIVE_INFINITY);
                }
                outputs[offset+row]=new Output(mixed[row].output(),hidden[row],logits);
            }
            sequence.position+=length;
            if(sequence.position==sequence.freshPrefillEnd)
                for(var attention:sequence.attention) if(attention!=null) attention.finishFreshPrefill();
            batchedChunks.incrementAndGet();
            batchedRows.addAndGet(length);
        } catch(IOException|RuntimeException|Error failure) {
            rollbacks.incrementAndGet();
            try {sequence.restore(before);} catch(IOException|RuntimeException|Error rollback) {failure.addSuppressed(rollback);}throw failure;
        } finally {before.close();}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var sequence:List.copyOf(sequences)) try {sequence.close();} catch(IOException closing) {
            if(failure==null) failure=closing;else failure.addSuppressed(closing);
        }
        for(var block:mtpResources) try {block.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        mtpResources.clear();
        for(var resource:hyperResources) try {resource.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        hyperResources.clear();
        for(var resource:moeResources) try {resource.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        moeResources.clear();
        for(var resource:pleResources) try {resource.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        pleResources.clear();
        if(pleReader!=null) try {pleReader.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        try {reader.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        if(failure!=null) throw failure;
    }
}

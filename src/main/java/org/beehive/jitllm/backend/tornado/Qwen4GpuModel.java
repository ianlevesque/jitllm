package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.cpu.Qwen4CpuEngine;
import org.beehive.jitllm.backend.cpu.Qwen4CpuModel;
import org.beehive.jitllm.backend.cpu.Qwen4MtpBlock;
import org.beehive.jitllm.backend.cpu.Qwen4SpeculativeSequence;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.ManagedDeviceModel;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.policy.*;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/** CUDA model adapter. Plans and packed weights are shared; recurrent/KV/PLE state is per sequence.
 * Activation vectors still cross the host between the independently owned device plans.
 */
public final class Qwen4GpuModel extends Qwen4CpuModel implements ManagedDeviceModel {
    private final FlashNextMatrixExecutor projections;
    private final FlashNextDeltaNetExecutor recurrence;
    private final FlashNextLayerExecutor layers;
    private final FlashNextQsaExecutor attention;
    private final FlashNextMoeExecutor moe;
    private final Qwen4MtpBlock mtp;
    private final FlashNextLayerExecutor mtpLayers;
    private final FlashNextQsaExecutor mtpAttention;
    private final FlashNextFp8MoeExecutor mtpExperts;
    private final FlashNextMtpFusionExecutor mtpFusion;
    private final Set<Qwen4SpeculativeSequence> speculative=Collections.newSetFromMap(new IdentityHashMap<>());
    private final List<AutoCloseable> resources;

    private Qwen4GpuModel(Qwen4CpuEngine engine, FlashNextMatrixExecutor projections,
            FlashNextDeltaNetExecutor recurrence, FlashNextLayerExecutor layers,
            FlashNextQsaExecutor attention, FlashNextMoeExecutor moe, Qwen4MtpBlock mtp,
            FlashNextLayerExecutor mtpLayers, FlashNextQsaExecutor mtpAttention,
            FlashNextFp8MoeExecutor mtpExperts, FlashNextMtpFusionExecutor mtpFusion, List<AutoCloseable> resources) {
        super(engine);this.projections=projections;this.recurrence=recurrence;this.layers=layers;
        this.attention=attention;this.moe=moe;this.resources=List.copyOf(resources);
        this.mtp=mtp;this.mtpLayers=mtpLayers;this.mtpAttention=mtpAttention;this.mtpExperts=mtpExperts;this.mtpFusion=mtpFusion;
    }

    public static Qwen4GpuModel load(Path directory, int contextLength, long pageBudget) throws IOException {
        var backend=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0);
        if (!backend.getBackendType().name().equals("CUDA"))
            throw new UnsupportedOperationException("Flash-Next NVFP4 device execution requires CUDA");
        // Preserve backend ordinals: swapping TornadoVM's default after initialization
        // can make an existing device ordinal refer to another GPU in task metadata.
        int device=Integer.getInteger("jitllm.qwen4.device",-1);
        if(device<0) {
            for(int i=0;i<backend.getNumDevices();i++) if(backend.getDevice(i).equals(backend.getDefaultDevice())) {device=i;break;}
        }
        if(device<0 || device>=backend.getNumDevices()) throw new IllegalArgumentException("Invalid Flash-Next CUDA device ordinal");
        // Cap raised from 16 GiB: on large cards the extra headroom goes to the MoE expert
        // cache (weights minus the separately capped projection budget). The device-memory
        // fraction bound is unchanged, so smaller cards resolve to the same defaults as before.
        long defaultWeights=Math.min(24L<<30,backend.getDevice(device).getMaxGlobalMemory()/2-(4L<<30));
        long weights=Long.getLong("jitllm.qwen4.gpuCacheBytes",defaultWeights);
        long projectionBudget=Long.getLong("jitllm.qwen4.gpuProjectionBytes",Math.min(8L<<30,weights*2/3));
        if(projectionBudget<=0 || projectionBudget>=weights) throw new IllegalArgumentException("Projection budget must fit inside the Flash-Next weight budget");
        if (weights<2) throw new IllegalArgumentException("Positive split Flash-Next weight cache budget required");
        int checkpoints=Integer.getInteger("jitllm.qwen4.checkpointCapacity",8);
        boolean useMtp=Boolean.getBoolean("jitllm.qwen4.mtp");
        if(useMtp && checkpoints<6) throw new IllegalArgumentException("MTP verification requires at least six checkpoint slots");
        var resources=new ArrayList<AutoCloseable>();
        try {
            var projections=new FlashNextMatrixExecutor(device,projectionBudget);resources.add(projections);
            var recurrence=new FlashNextDeltaNetExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuStateBytes",4L<<30));resources.add(recurrence);
            var layers=new FlashNextLayerExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuLayerBytes",64L<<20));resources.add(layers);
            var attention=new FlashNextQsaExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuQsaBytes",16L<<30),true);resources.add(attention);
            var moe=new FlashNextMoeExecutor(device,weights-projectionBudget);resources.add(moe);
            var engine=new Qwen4CpuEngine(directory,contextLength,pageBudget,null,projections,recurrence,layers,attention,moe);
            resources.add(engine);
            Qwen4MtpBlock mtp=null;FlashNextLayerExecutor mtpLayers=null;FlashNextQsaExecutor mtpAttention=null;
            FlashNextFp8MoeExecutor mtpExperts=null;FlashNextMtpFusionExecutor mtpFusion=null;
            if(useMtp) {
                mtpLayers=new FlashNextLayerExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuMtpLayerBytes",64L<<20),true);resources.add(mtpLayers);
                mtpAttention=new FlashNextQsaExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuMtpQsaBytes",2L<<30),true);resources.add(mtpAttention);
                mtpExperts=new FlashNextFp8MoeExecutor(device,Long.getLong("jitllm.qwen4.gpuMtpExpertBytes",128L<<20));resources.add(mtpExperts);
                mtpFusion=new FlashNextMtpFusionExecutor(device,Long.getLong("jitllm.qwen4.gpuMtpFusionBytes",128L<<20));resources.add(mtpFusion);
                mtp=engine.newMtpBlock(checkpoints,projections,mtpLayers,mtpAttention,mtpExperts,mtpFusion);
            }
            return new Qwen4GpuModel(engine,projections,recurrence,layers,attention,moe,mtp,mtpLayers,mtpAttention,mtpExperts,mtpFusion,resources);
        } catch(IOException | RuntimeException | Error failure) {
            closeAll(resources,failure);throw failure;
        }
    }

    @Override public synchronized State createNewState(int batchsize) {
        requireOpen();if(mtp==null) return super.createNewState(batchsize);
        return State.withPrefillBatchSize(1,()->{
            try {
                var holder=new Qwen4SpeculativeSequence[1];
                var sequence=Qwen4SpeculativeSequence.create(engine(),mtp,()->{synchronized(Qwen4GpuModel.this) {speculative.remove(holder[0]);}});
                holder[0]=sequence;speculative.add(sequence);
                try {var state=new Qwen4State(configuration(),batchsize,sequence);state.latestToken=-1;return state;}
                catch(RuntimeException|Error failure) {try {sequence.close();} catch(IOException closing) {failure.addSuppressed(closing);}throw failure;}
            } catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}
        });
    }

    @Override public StorageOptions defaultDeviceStorage() { return StorageOptions.fp8(); }
    @Override public void validateDeviceOptions(ExecutionPolicy policy, StorageOptions storage) {
        if (storage.keyValueRepresentation()!=DataType.F8_E4M3 || storage.sharedKeyValuePool())
            throw new UnsupportedOperationException("Flash-Next CUDA uses session-owned FP8 KV pages; select StorageOptions.fp8()");
        if (policy.prefillBatchSize()!=1 || policy.samplingResidency()!=ExecutionPolicy.SamplingResidency.HOST
                || policy.splitKvPartitions().isPresent() || policy.packedHalf2Attention() || policy.scalarFp16KeyValueReads())
            throw new UnsupportedOperationException("Flash-Next uses model-owned prefill (sequential, or distinct-row batched under jitllm.qwen4.batchedPrefill) and host sampling; policy prefill batching and legacy attention overrides are not implemented; use jitllm.qwen4.prefillChunk for chunked prefill geometry");
    }
    @Override public synchronized ExecutionInfo prepareDevice(State state) {
        requireOpen();
        if (!(state instanceof org.beehive.jitllm.inference.state.Qwen4State)) throw new IllegalArgumentException("Expected Flash-Next state");
        String prefill=engine().batchedPrefillEnabled()
                ? "batched prefill (up to "+Qwen4CpuEngine.BatchedPrefill.effectiveRows(engine().batchedPrefillRows())+" rows)"
                : "sequential prefill";
        return new ExecutionInfo("CUDA",projections.deviceName(),"TornadoVM / CUDA", mtp==null ? prefill+" and decode" : prefill+" / MTP draft-three target verification",1,
                "FP8 E4M3 paged KV / BF16 fresh-prefill shadow", "BF16 projections / native NVFP4 experts", "QSA sparse GQA",false,true,
                "cuBLASLt NVFP4 experts / FP32 HC; cuBLAS BF16 HC prefill; FlashInfer XQA decode"+(mtp==null ? "" : "; native block-FP8 MTP / target sampling")
                        +(engine().prefillChunk()>1 ? "; prefill geometry chunk "+engine().prefillChunk() : ""));
    }
    private void requireOpen() { if(engine().isClosed()) throw new IllegalStateException("Flash-Next CUDA model is closed"); }
    /** Shared HC/projection/MoE workspaces require serialization across sessions. */
    @Override public synchronized List<Integer> generateOnDevice(State state,int position,List<Integer> prompt,
            Set<Integer> stops,int budget,Sampler sampler,IntConsumer callback) {
        requireOpen();
        try {
            if(state instanceof Qwen4State qwen && qwen.sequence() instanceof Qwen4SpeculativeSequence sequence) {
                if(!speculative.contains(sequence)) throw new IllegalArgumentException("Speculative state belongs to a different model");
                var generated=sequence.generate(position,prompt,stops,budget,sampler,callback);
                if(!generated.isEmpty()) state.latestToken=generated.getLast();return generated;
            }
            return super.generateTokens(state,position,prompt,stops,budget,sampler,false,callback);
        } catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}
        catch(RuntimeException | Error failure) {
            failure.addSuppressed(new IllegalStateException("Flash-Next resource accounting: "+deviceStatistics()+"; native="+FlashNextNvfp4Library.statistics()));
            throw failure;
        }
    }
    @Override public List<Integer> generateTokensGPU(State state,int position,List<Integer> prompt,Set<Integer> stops,
            int budget,Sampler sampler,boolean echo,IntConsumer callback,TornadoVMMasterPlan ignored) {
        return generateOnDevice(state,position,prompt,stops,budget,sampler,callback);
    }
    @Override public List<Integer> generateTokens(State state,int position,List<Integer> prompt,Set<Integer> stops,
            int budget,Sampler sampler,boolean echo,IntConsumer callback) {
        return generateOnDevice(state,position,prompt,stops,budget,sampler,callback);
    }
    public Map<String,Object> deviceStatistics() {
        var p=projections.statistics();var r=recurrence.statistics();var l=layers.statistics();var q=attention.statistics();var m=moe.statistics();
        var result=new LinkedHashMap<String,Object>();
        result.put("projection_bytes",p.residentBytes());result.put("projection_executions",p.executions());result.put("bf16_gemm_bytes",p.bf16ResidentBytes());
        result.put("recurrence_bytes",r.residentBytes());result.put("recurrence_sessions",r.sessions());result.put("recurrence_forwards",r.forwards());
        result.put("recurrence_snapshots",r.liveSnapshots());result.put("layer_bytes",l.residentBytes());result.put("ple_sessions",l.pleSessions());result.put("layer_snapshots",l.liveSnapshots());
        result.put("qsa_reserved_bytes",q.reservedBytes());result.put("qsa_materialized_bytes",q.materializedPageBytes());result.put("qsa_sessions",q.sessions());result.put("qsa_snapshots",q.liveSnapshots());
        result.put("moe_bytes",m.residentBytes());result.put("moe_layers",m.layers());result.put("moe_forwards",m.forwards());result.put("moe_evictions",m.evictions());
        result.put("moe_intermediate_download_bytes",m.intermediateDownloadBytes());
        var e=engine().forwardStatistics();var bf=projections.bf16Statistics();var ff=projections.bf16FloatStatistics();
        result.put("prefill_chunk",engine().prefillChunk());
        var bp=engine().batchedPrefillStatistics();
        result.put("batched_prefill_enabled",engine().batchedPrefillEnabled());
        result.put("batched_prefill_rows",engine().batchedPrefillRows());
        result.put("batched_prefill_chunks",bp.chunks());result.put("batched_prefill_ingested_rows",bp.rows());
        result.put("batched_prefill_fallbacks",bp.fallbacks());
        result.put("batched_prefill_embedding_rows",bp.embeddingRows());result.put("batched_prefill_ple_rows",bp.pleRows());
        result.put("batched_prefill_hyper_rows",bp.hyperRows());result.put("batched_prefill_delta_rows",bp.deltaRows());
        result.put("batched_prefill_qsa_rows",bp.qsaRows());result.put("batched_prefill_moe_rows",bp.moeRows());
        result.put("engine_forwards",e.forwards());result.put("engine_prefill_forwards",e.prefillForwards());result.put("engine_rollbacks",e.rollbacks());
        result.put("projection_uploads",p.uploads());result.put("projection_evictions",p.evictions());
        result.put("projection_cache_hits",p.cacheHits());result.put("projection_cache_misses",p.cacheMisses());
        result.put("bf16_gemm_uploads",bf.uploads());result.put("bf16_gemm_evictions",bf.evictions());result.put("bf16_gemm_executions",bf.executions());
        result.put("bf16_float_bytes",ff.residentBytes());result.put("bf16_float_uploads",ff.uploads());result.put("bf16_float_evictions",ff.evictions());result.put("bf16_float_executions",ff.executions());
        result.put("moe_expert_cache_hits",m.expertCacheHits());result.put("moe_expert_cache_misses",m.expertCacheMisses());
        result.put("moe_expert_file_bytes",m.expertFileBytes());result.put("moe_expert_evicted_bytes",m.expertEvictedBytes());result.put("moe_registered_experts",m.registeredExperts());
        result.put("host_upload_bytes",p.hostUploadBytes()+r.hostUploadBytes()+l.hostUploadBytes()+q.hostUploadBytes()+m.hostUploadBytes());
        result.put("host_download_bytes",p.hostDownloadBytes()+r.hostDownloadBytes()+l.hostDownloadBytes()+q.hostDownloadBytes()+m.hostDownloadBytes());
        result.put("mtp_enabled",mtp!=null);result.put("mtp_sessions",speculative.size());
        if(mtp!=null) {
            var ml=mtpLayers.statistics();var mq=mtpAttention.statistics();var me=mtpExperts.statistics();var mf=mtpFusion.statistics();
            result.put("mtp_layer_bytes",ml.residentBytes());result.put("mtp_qsa_reserved_bytes",mq.reservedBytes());result.put("mtp_qsa_snapshots",mq.liveSnapshots());
            result.put("mtp_expert_bytes",me.residentBytes());result.put("mtp_expert_forwards",me.forwards());result.put("mtp_fusion_bytes",mf.residentBytes());result.put("mtp_fusion_forwards",mf.forwards());
            result.put("mtp_kv_download_bytes",mq.kvHostDownloadBytes());
            result.put("mtp_expert_cache_hits",me.expertCacheHits());result.put("mtp_expert_cache_misses",me.expertCacheMisses());result.put("mtp_expert_file_bytes",me.expertFileBytes());
        }
        return result;
    }
    private static void closeAll(List<AutoCloseable> resources,Throwable failure) {
        // Engine-owned request states and blocks borrow the executors. Release them first.
        for(var resource:resources) if(resource instanceof Qwen4CpuEngine) try {resource.close();} catch(Exception closing) {failure.addSuppressed(closing);}
        for(int i=resources.size()-1;i>=0;i--) if(!(resources.get(i) instanceof Qwen4CpuEngine))
            try {resources.get(i).close();} catch(Exception closing) {failure.addSuppressed(closing);}
    }
    @Override public synchronized void close() throws IOException {
        IOException sequenceFailure=null;
        for(var sequence:List.copyOf(speculative)) try {sequence.close();} catch(IOException failure) {
            if(sequenceFailure==null) sequenceFailure=failure;else sequenceFailure.addSuppressed(failure);
        }
        var failure=new IOException("Cannot close Flash-Next CUDA resources");closeAll(resources,failure);
        if(sequenceFailure!=null) failure.addSuppressed(sequenceFailure);
        if(failure.getSuppressed().length!=0) throw failure;
    }
}

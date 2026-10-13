package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.cpu.Qwen4CpuEngine;
import org.beehive.jitllm.backend.cpu.Qwen4CpuModel;
import org.beehive.jitllm.inference.sampler.Sampler;
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
    private final List<AutoCloseable> resources;

    private Qwen4GpuModel(Qwen4CpuEngine engine, FlashNextMatrixExecutor projections,
            FlashNextDeltaNetExecutor recurrence, FlashNextLayerExecutor layers,
            FlashNextQsaExecutor attention, FlashNextMoeExecutor moe, List<AutoCloseable> resources) {
        super(engine);this.projections=projections;this.recurrence=recurrence;this.layers=layers;
        this.attention=attention;this.moe=moe;this.resources=List.copyOf(resources);
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
        long defaultWeights=Math.min(16L<<30,backend.getDevice(device).getMaxGlobalMemory()/2-(4L<<30));
        long weights=Long.getLong("jitllm.qwen4.gpuCacheBytes",defaultWeights);
        long projectionBudget=Long.getLong("jitllm.qwen4.gpuProjectionBytes",Math.min(8L<<30,weights*2/3));
        if(projectionBudget<=0 || projectionBudget>=weights) throw new IllegalArgumentException("Projection budget must fit inside the Flash-Next weight budget");
        if (weights<2) throw new IllegalArgumentException("Positive split Flash-Next weight cache budget required");
        int checkpoints=Integer.getInteger("jitllm.qwen4.checkpointCapacity",8);
        var resources=new ArrayList<AutoCloseable>();
        try {
            var projections=new FlashNextMatrixExecutor(device,projectionBudget);resources.add(projections);
            var recurrence=new FlashNextDeltaNetExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuStateBytes",4L<<30));resources.add(recurrence);
            var layers=new FlashNextLayerExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuLayerBytes",64L<<20));resources.add(layers);
            var attention=new FlashNextQsaExecutor(device,checkpoints,Long.getLong("jitllm.qwen4.gpuQsaBytes",16L<<30),true);resources.add(attention);
            var moe=new FlashNextMoeExecutor(device,weights-projectionBudget);resources.add(moe);
            var engine=new Qwen4CpuEngine(directory,contextLength,pageBudget,null,projections,recurrence,layers,attention,moe);
            resources.add(engine);
            return new Qwen4GpuModel(engine,projections,recurrence,layers,attention,moe,resources);
        } catch(IOException | RuntimeException | Error failure) {
            closeAll(resources,failure);throw failure;
        }
    }

    @Override public StorageOptions defaultDeviceStorage() { return StorageOptions.fp8(); }
    @Override public void validateDeviceOptions(ExecutionPolicy policy, StorageOptions storage) {
        if (storage.keyValueRepresentation()!=DataType.F8_E4M3 || storage.sharedKeyValuePool())
            throw new UnsupportedOperationException("Flash-Next CUDA uses session-owned FP8 KV pages; select StorageOptions.fp8()");
        if (policy.prefillBatchSize()!=1 || policy.samplingResidency()!=ExecutionPolicy.SamplingResidency.HOST
                || policy.splitKvPartitions().isPresent() || policy.packedHalf2Attention() || policy.scalarFp16KeyValueReads())
            throw new UnsupportedOperationException("Flash-Next currently uses sequential prefill and host sampling; batched prefill and legacy attention overrides are not implemented");
    }
    @Override public synchronized ExecutionInfo prepareDevice(State state) {
        requireOpen();
        if (!(state instanceof org.beehive.jitllm.inference.state.Qwen4State)) throw new IllegalArgumentException("Expected Flash-Next state");
        return new ExecutionInfo("CUDA",projections.deviceName(),"TornadoVM / CUDA", "sequential prefill and decode",1,
                "FP8 E4M3 paged KV / BF16 fresh-prefill shadow", "BF16 projections / native NVFP4 experts", "QSA sparse GQA",false,true,
                "cuBLASLt NVFP4 experts / FP32 HC; cuBLAS BF16 HC prefill; FlashInfer XQA decode");
    }
    private void requireOpen() { if(engine().isClosed()) throw new IllegalStateException("Flash-Next CUDA model is closed"); }
    /** Shared HC/projection/MoE workspaces require serialization across sessions. */
    @Override public synchronized List<Integer> generateOnDevice(State state,int position,List<Integer> prompt,
            Set<Integer> stops,int budget,Sampler sampler,IntConsumer callback) {
        requireOpen();
        try {return super.generateTokens(state,position,prompt,stops,budget,sampler,false,callback);}
        catch(RuntimeException | Error failure) {
            failure.addSuppressed(new IllegalStateException("Flash-Next resource accounting: "+deviceStatistics()+"; native="+FlashNextNvfp4Library.statistics()));
            throw failure;
        }
    }
    @Override public List<Integer> generateTokensGPU(State state,int position,List<Integer> prompt,Set<Integer> stops,
            int budget,Sampler sampler,boolean echo,IntConsumer callback,TornadoVMMasterPlan ignored) {
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
        result.put("moe_intermediate_download_bytes",m.intermediateDownloadBytes());return result;
    }
    private static void closeAll(List<AutoCloseable> resources,Throwable failure) {
        for(int i=resources.size()-1;i>=0;i--) try {resources.get(i).close();} catch(Exception closing) {failure.addSuppressed(closing);}
    }
    @Override public synchronized void close() throws IOException {
        var failure=new IOException("Cannot close Flash-Next CUDA resources");closeAll(resources,failure);
        if(failure.getSuppressed().length!=0) throw failure;
    }
}

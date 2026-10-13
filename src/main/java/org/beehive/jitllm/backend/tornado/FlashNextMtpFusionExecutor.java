package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.inference.op.MtpFusionExecutor;
import org.beehive.jitllm.backend.tornado.kernels.*;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.CuBlasOptions;

/** Native BF16 MTP fusion: global norms, two GEMMs and residual addition in one graph.
 * Caller-owned cuBLAS plans retain their workspace across stream changes.
 */
public final class FlashNextMtpFusionExecutor implements MtpFusionExecutor,AutoCloseable {
    private static final long WORKSPACE=32L<<20;
    public record Statistics(long residentBytes,long peakBytes,long builds,long evictions,long forwards,long kernelNanos) {}
    private final TornadoDevice device;
    private final long budget;
    private final Set<DeviceFusion> fusions=Collections.newSetFromMap(new IdentityHashMap<>());
    private final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,peak,builds,evictions,forwards,kernelNanos,graphId;
    private boolean closed;
    private record Key(DeviceFusion fusion,int rows,long thread) {}
    public FlashNextMtpFusionExecutor(int deviceIndex,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive MTP fusion budget required");
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);this.budget=budget;
    }
    public synchronized Statistics statistics() {return new Statistics(resident,peak,builds,evictions,forwards,kernelNanos);}
    @Override public synchronized Fusion newFusion(ModelOptMatrix embedding,ModelOptMatrix hidden,int branches,float epsilon,float[] en,float[] hn) {
        if(closed) throw new IllegalStateException("MTP fusion executor is closed");
        int dim=embedding.rows();
        if(!embedding.dtype().equals("BF16") || !hidden.dtype().equals("BF16") || embedding.columns()!=dim || hidden.rows()!=dim || hidden.columns()!=dim
                || dim%8!=0 || branches<2 || en.length!=dim || hn.length!=Math.multiplyExact(dim,branches) || !Float.isFinite(epsilon) || epsilon<=0)
            throw new IllegalArgumentException("Invalid shared-branch MTP fusion");
        var fusion=new DeviceFusion(embedding,hidden,branches,epsilon,en.clone(),hn.clone());fusions.add(fusion);return fusion;
    }
    private final class DeviceFusion implements Fusion {
        final ModelOptMatrix embedding,hidden;final int dim,branches;final float epsilon;final float[] en,hn;
        boolean released;
        DeviceFusion(ModelOptMatrix e,ModelOptMatrix h,int branches,float epsilon,float[] en,float[] hn) {
            embedding=e;hidden=h;dim=e.rows();this.branches=branches;this.epsilon=epsilon;this.en=en;this.hn=hn;
        }
        @Override public float[] forward(float[] embeddings,float[] hidden,int rows) throws IOException {
            synchronized(FlashNextMtpFusionExecutor.this) {
                if(released || closed) throw new IOException("MTP fusion is closed");
                if(rows<=0 || embeddings.length!=Math.multiplyExact(rows,dim) || hidden.length!=Math.multiplyExact(rows,dim*branches))
                    throw new IllegalArgumentException("Invalid MTP fusion batch");
                for(float v:embeddings) if(!Float.isFinite(v)) throw new IllegalArgumentException("Nonfinite embedding");
                for(float v:hidden) if(!Float.isFinite(v)) throw new IllegalArgumentException("Nonfinite HC state");
                var key=new Key(this,rows,Thread.currentThread().threadId());var entry=cache.get(key);
                if(entry==null) {
                    long size=size(key);if(size>budget) throw new IOException("MTP fusion exceeds cache budget "+budget);
                    while(size>budget-resident) {
                        var iterator=cache.entrySet().iterator();var oldest=iterator.next();oldest.getValue().close();iterator.remove();resident-=oldest.getValue().bytes;evictions++;
                    }
                    entry=new Entry(key);cache.put(key,entry);resident+=entry.bytes;peak=Math.max(peak,resident);builds++;
                }
                for(int i=0;i<embeddings.length;i++) entry.embeddingInput.set(i,FloatFormats.roundBf16(embeddings[i]));
                for(int i=0;i<hidden.length;i++) entry.hiddenInput.set(i,FloatFormats.roundBf16(hidden[i]));
                try {
                    entry.last=entry.plan.execute();long time=entry.last.getProfilerResult().getDeviceKernelTime();if(time<=0) throw new IOException("MTP fusion recorded no CUDA kernels");
                    kernelNanos+=time;forwards++;float[] result=new float[hidden.length];for(int i=0;i<result.length;i++) result[i]=entry.output.get(i);return result;
                } catch(RuntimeException failure) {throw new IOException("Native MTP fusion failed",failure);}
            }
        }
        @Override public void close() throws IOException {
            synchronized(FlashNextMtpFusionExecutor.this) {
                if(released) return;released=true;IOException failure=null;
                var iterator=cache.entrySet().iterator();while(iterator.hasNext()) {var item=iterator.next();if(item.getKey().fusion==this) {
                    try {item.getValue().close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
                    resident-=item.getValue().bytes;iterator.remove();}}
                fusions.remove(this);if(failure!=null) throw failure;
            }
        }
    }
    private static long size(Key key) {
        var f=key.fusion;long e=(long)key.rows*f.dim,h=e*f.branches;
        return f.embedding.weights().bytes()+f.hidden.weights().bytes()+12*e+16*h+4L*(f.dim+f.dim*f.branches)+13L*TornadoNativeArray.ARRAY_HEADER+WORKSPACE;
    }
    public synchronized Map<String,float[]> intermediates(Fusion fusion,int rows) {
        if(!(fusion instanceof DeviceFusion resource) || !fusions.contains(resource)) throw new IllegalArgumentException("Unknown MTP fusion");
        var entry=cache.get(new Key(resource,rows,Thread.currentThread().threadId()));if(entry==null || entry.last==null) throw new IllegalStateException("No current fusion execution");
        entry.last.transferToHost(entry.normalizedE,entry.normalizedH,entry.projectedE,entry.projectedH);
        var result=new LinkedHashMap<String,float[]>();
        for(var pair:List.of(Map.entry("normalized_embedding",entry.normalizedE),Map.entry("normalized_hidden",entry.normalizedH))) {
            float[] values=new float[pair.getValue().getSize()];for(int i=0;i<values.length;i++) values[i]=pair.getValue().get(i);result.put(pair.getKey(),values);
        }
        for(var pair:List.of(Map.entry("projected_embedding",entry.projectedE),Map.entry("projected_hidden",entry.projectedH))) {
            float[] values=new float[pair.getValue().getSize()];for(int i=0;i<values.length;i++) values[i]=Float.intBitsToFloat((pair.getValue().get(i)&65535)<<16);result.put(pair.getKey(),values);
        }return result;
    }
    private static LibraryTaskDescriptor gemm(int m,int n,int k,BFloat16Array weights,BFloat16Array input,BFloat16Array output) {
        return CuBlas.cublasGemmExBF16(1,0,m,n,k,1f,weights,k,input,k,0f,output,m).withTuning(new CuBlasOptions().withWorkspace(WORKSPACE));
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();final long bytes;
        final FloatArray embeddingInput,hiddenInput,output,normalizedE,normalizedH;final BFloat16Array projectedE,projectedH;final TornadoExecutionPlan plan;
        TornadoExecutionResult last;
        Entry(Key key) throws IOException {
            bytes=size(key);var f=key.fusion;int e=Math.multiplyExact(key.rows,f.dim),h=Math.multiplyExact(e,f.branches);TornadoExecutionPlan built=null;
            try {
                embeddingInput=floats(e);hiddenInput=floats(h);output=floats(h);
                var en=floats(f.en);var hn=floats(f.hn);normalizedE=floats(e);normalizedH=floats(h);
                var packedE=bfloats(e);var packedH=bfloats(h);projectedE=bfloats(e);projectedH=bfloats(h);
                var ew=weights(f.embedding);var hw=weights(f.hidden);
                String graphName="flashnextMtpFusion"+(graphId++);
                var graph=new TaskGraph(graphName)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,ew,hw,en,hn,packedE,packedH,projectedE,projectedH)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,embeddingInput,hiddenInput)
                        .libraryTask("embeddingNorm",FlashNextRmsNormLibrary::normalize,key.rows,f.dim,f.epsilon,embeddingInput,en,normalizedE)
                        .libraryTask("hiddenNorm",FlashNextRmsNormLibrary::normalize,key.rows,f.dim*f.branches,f.epsilon,hiddenInput,hn,normalizedH)
                        .task("packEmbedding",FlashNextWeightKernels::bf16GemmInput,normalizedE,packedE,f.dim,0)
                        .task("packHidden",FlashNextWeightKernels::bf16GemmInput,normalizedH,packedH,f.dim,0)
                        .libraryTask("embeddingProjection",FlashNextMtpFusionExecutor::gemm,f.dim,key.rows,f.dim,ew,packedE,projectedE)
                        .libraryTask("hiddenProjection",FlashNextMtpFusionExecutor::gemm,f.dim,key.rows*f.branches,f.dim,hw,packedH,projectedH)
                        .task("fuse",FlashNextMtpKernels::fuse,projectedE,projectedH,output,f.dim,f.branches)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output).transferToHost(DataTransferMode.UNDER_DEMAND,normalizedE,normalizedH,projectedE,projectedH);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT);plan=built;
            } catch(Exception|Error failure) {if(built!=null) try {built.close();} catch(Exception closing) {failure.addSuppressed(closing);}arena.close();throw new IOException("Cannot build MTP fusion",failure);}
        }
        FloatArray floats(int n) {return FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*n,32));}
        FloatArray floats(float[] values) {var out=floats(values.length);for(int i=0;i<values.length;i++) out.set(i,values[i]);return out;}
        BFloat16Array bfloats(int n) {return BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+2L*n,32));}
        BFloat16Array weights(ModelOptMatrix matrix) throws IOException {
            var out=bfloats(Math.toIntExact(matrix.weights().elements()));
            try(var temporary=Arena.ofConfined();var file=FileChannel.open(matrix.weights().file(),StandardOpenOption.READ)) {
                out.getSegment().copyFrom(file.map(FileChannel.MapMode.READ_ONLY,matrix.weights().offset(),matrix.weights().bytes(),temporary));
            }return out;
        }
        @Override public void close() throws IOException {try {plan.close();} catch(Exception failure) {throw new IOException("Cannot close MTP fusion",failure);} finally {arena.close();}}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;IOException failure=null;
        for(var fusion:List.copyOf(fusions)) try {fusion.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        closed=true;if(failure!=null) throw failure;
    }
}

package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.kernels.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.inference.op.MoeExecutor;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Complete device MoE decode layer: BF16 router/shared expert, CUDA routing/encoding,
 * packed native selected experts and finalization. Only selected IDs and final outputs cross back.
 * Compute plans/workspaces are per layer; expert residency is an independent bounded LRU.
 */
public final class FlashNextMoeExecutor implements MoeExecutor,AutoCloseable {
    public record Statistics(long residentBytes,long layerBytes,long expertBytes,int layers,int registeredExperts,long uploads,long evictions,
                             long expertFileBytes,long forwards,long kernelNanos,long idDownloadBytes,long intermediateDownloadBytes,long peakBytes,long layerBuilds,
                             long expertCacheHits,long expertCacheMisses,long expertEvictedBytes,long hostUploadBytes,long hostDownloadBytes) {}
    private static final int PAD=32;
    private final TornadoDevice device;
    private final long budget;
    private final Set<DeviceLayer> layers=Collections.newSetFromMap(new IdentityHashMap<>());
    private final FlashNextExpertCache weights;
    private long layerBytes,graphId,forwards,kernelNanos,idDownloads,peakBytes,layerBuilds,hostUploadBytes,hostDownloadBytes;
    private boolean closed;
    public FlashNextMoeExecutor(int index,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive MoE budget required");this.budget=budget;
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(index);weights=new FlashNextExpertCache(device,budget);
    }
    public synchronized Statistics statistics() {return new Statistics(layerBytes+weights.resident,layerBytes,weights.resident,layers.size(),weights.cache.size(),weights.uploads,weights.evictions,weights.readBytes,forwards,kernelNanos,idDownloads,0,peakBytes,layerBuilds,weights.hits,weights.misses,weights.evictedBytes,hostUploadBytes,hostDownloadBytes);}
    @Override public synchronized Layer newLayer(ModelOptMatrix router,ModelOptMatrix sharedGate,ModelOptMatrix sharedUp,ModelOptMatrix sharedDown,ModelOptMatrix sharedRouter,
                                                ModelOptMatrix[] gates,ModelOptMatrix[] ups,ModelOptMatrix[] downs,int topK,boolean normalize,float inputScale,float intermediateScale) throws IOException {
        if(closed) throw new IllegalStateException("MoE executor is closed");
        if(router.rows()!=512 || gates.length!=512 || ups.length!=512 || downs.length!=512 || topK<=0 || topK>512
                || !Float.isFinite(inputScale) || inputScale<=0 || !Float.isFinite(intermediateScale) || intermediateScale<=0)
            throw new IllegalArgumentException("Invalid GPU MoE geometry/scales");
        for(int i=0;i<512;i++) if(!gates[i].dtype().equals("U8") || !ups[i].dtype().equals("U8") || !downs[i].dtype().equals("U8")
                || gates[i].globalScale()!=ups[i].globalScale() || gates[i].rows()!=ups[i].rows() || gates[i].columns()!=ups[i].columns())
            throw new IOException("Combined NVFP4 up/gate layout requires matching shapes/alphas at expert "+i);
        var layer=new DeviceLayer(router,sharedGate,sharedUp,sharedDown,sharedRouter,gates,ups,downs,topK,normalize,inputScale,intermediateScale);
        if(layer.bytes>budget-layerBytes-weights.resident) {layer.close();throw new IOException("MoE layer allocation exceeds budget");}
        layers.add(layer);layerBytes+=layer.bytes;weights.budget=budget-layerBytes;layerBuilds++;peakBytes=Math.max(peakBytes,layerBytes+weights.resident);return layer;
    }
    private final class DeviceLayer implements Layer {
        final Arena arena=Arena.ofShared();final ModelOptMatrix[] gates,ups,downs;
        final int dim,hidden,top;
        final float inputScale,midScale;
        final FloatArray input,output,routeWeights,diagnosticWeights,alpha1,alpha2;
        final IntArray selected,resources1,resources2;
        final TornadoExecutionPlan plan;
        long bytes;boolean released;
        DeviceLayer(ModelOptMatrix router,ModelOptMatrix sg,ModelOptMatrix su,ModelOptMatrix sd,ModelOptMatrix sr,
                    ModelOptMatrix[] gates,ModelOptMatrix[] ups,ModelOptMatrix[] downs,int top,boolean normalize,float inputScale,float midScale) throws IOException {
            this.gates=gates.clone();this.ups=ups.clone();this.downs=downs.clone();this.top=top;this.inputScale=inputScale;this.midScale=midScale;
            dim=router.columns();hidden=gates[0].rows();TornadoExecutionPlan built=null;
            try {
                if(dim%64!=0 || hidden%128!=0) throw new IOException("Unsupported packed MoE dimensions");
                var rw=dense(router);var gw=dense(sg);var uw=dense(su);var dw=dense(sd);var sw=dense(sr);
                input=floats(dim);output=floats(dim);selected=ints(top);routeWeights=floats(top);diagnosticWeights=floats(top);
                var logits=floats(512);var sharedGate=floats(sg.rows());var sharedUp=floats(su.rows());var sharedMid=floats(sg.rows());var scratch=floats(1);
                var shared=floats(dim);var sharedRouter=floats(1);
                var padded=floats(PAD*dim);var packed=bytes(PAD*dim/2);var rawSf=bytes(PAD*dim/16);var sf=bytes(128*dim/16);
                var mid=floats(top*PAD*hidden);var midPacked=bytes(top*PAD*hidden/2);var midRawSf=bytes(top*PAD*hidden/16);var midSf=bytes(top*128*hidden/16);
                var upGate=bfloats(top*PAD*2*hidden);var down=bfloats(top*PAD*dim);
                var e4=floats(256);var e2=floats(8);for(int i=0;i<256;i++) e4.set(i,FloatFormats.e4m3(i));for(int i=0;i<8;i++) e2.set(i,FloatFormats.e2m1(i));
                var mode=ints(5);mode.set(1,1);mode.set(2,127);mode.set(3,8);mode.set(4,2);
                var cal1=floats(3);cal1.set(2,1f/inputScale);var cal2=floats(3);cal2.set(2,1f/midScale);
                resources1=ints(top);resources2=ints(top);alpha1=floats(2*top);alpha2=floats(2*top);
                String id="flashnextMoe"+(graphId++);
                var preparation=new TaskGraph(id)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,rw,gw,uw,dw,sw,padded,e4,e2,mode,cal1,cal2)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,input)
                        .task("router",FlashNextWeightKernels::denseMatvec,input,rw,logits,dim,0).task("routerBoundary",FlashNextMoeKernels::bf16Boundary,logits)
                        .task("gate",FlashNextWeightKernels::denseMatvec,input,gw,sharedGate,dim,0).task("gateBoundary",FlashNextMoeKernels::bf16Boundary,sharedGate)
                        .task("up",FlashNextWeightKernels::denseMatvec,input,uw,sharedUp,dim,0).task("upBoundary",FlashNextMoeKernels::bf16Boundary,sharedUp)
                        .task("sharedActivation",FlashNextMoeKernels::swiglu,sharedGate,sharedUp,sharedMid)
                        .task("sharedDown",FlashNextWeightKernels::denseMatvec,sharedMid,dw,shared,sg.rows(),0).task("sharedBoundary",FlashNextMoeKernels::bf16Boundary,shared)
                        .task("sharedRouter",FlashNextWeightKernels::denseMatvec,input,sw,sharedRouter,dim,0).task("sharedRouterBoundary",FlashNextMoeKernels::bf16Boundary,sharedRouter)
                        .task("route",FlashNextMoeKernels::route,new KernelContext(),logits,selected,routeWeights,normalize ? 1 : 0)
                        .task("pad",FlashNextMoeExecutor::pad,input,padded)
                        .task("seedEncoding",FlashNextMoeKernels::seedEncoding,e4,e2,mode,cal1,cal2,scratch)
                        .persistOnDevice(padded,routeWeights,shared,sharedRouter,e4,e2,mode,cal1,cal2)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,selected,routeWeights);
                var fc1=new TaskGraph(id+"Fc1").consumeFromDevice(id,padded,e4,e2,mode,cal1)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,packed,rawSf,sf,upGate)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,resources1,alpha1)
                        .task("encode",FlashNextWeightKernels::quantizeNvfp4,padded,cal1,mode,e4,e2,packed,rawSf)
                        .task("scaleLayout",FlashNextWeightKernels::swizzleNvfp4Scales,rawSf,sf,mode,PAD,dim/16,0)
                        .libraryTask("experts",FlashNextNvfp4Library::experts,2*hidden,PAD,dim,top,resources1,packed,sf,alpha1,upGate,0,0)
                        .task("activation",FlashNextMoeKernels::expertActivation,upGate,mid,hidden,PAD).persistOnDevice(mid);
                var fc2=new TaskGraph(id+"Fc2").consumeFromDevice(id,e4,e2,mode,cal2,routeWeights,shared,sharedRouter).consumeFromDevice(id+"Fc1",mid)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,midPacked,midRawSf,midSf,down)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,resources2,alpha2)
                        .task("encode",FlashNextWeightKernels::quantizeNvfp4,mid,cal2,mode,e4,e2,midPacked,midRawSf)
                        .task("scaleLayout",FlashNextMoeKernels::swizzleExperts,midRawSf,midSf,hidden/16,PAD)
                        .libraryTask("experts",FlashNextNvfp4Library::experts,dim,PAD,hidden,top,resources2,midPacked,midSf,alpha2,down,PAD*hidden/2,128*hidden/16)
                        .task("finish",FlashNextMoeKernels::finish,down,routeWeights,shared,sharedRouter,output,PAD)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output);
                var diagnostic=new TaskGraph(id+"Weights").consumeFromDevice(id,routeWeights)
                        .task("read",FlashNextMoeKernels::readWeights,routeWeights,diagnosticWeights).transferToHost(DataTransferMode.EVERY_EXECUTION,diagnosticWeights);
                var worker=new WorkerGrid1D(512);worker.setLocalWork(512,1,1);var scheduler=new GridScheduler(id+".route",worker);
                bytes+=FlashNextNvfp4Library.WORKSPACE_BYTES;
                built=new TornadoExecutionPlan(preparation.snapshot(),fc1.snapshot(),fc2.snapshot(),diagnostic.snapshot()).withDevice(device).withGridScheduler(scheduler)
                        .withProfiler(ProfilerMode.SILENT).withCompilerFlags(TornadoVMBackendType.CUDA,"--prec-div=false");plan=built;
            } catch(Exception|Error error) {if(built!=null) try {built.close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}arena.close();throw new IOException("Cannot construct GPU MoE layer",error);}
        }
        FloatArray floats(int n) {long size=TornadoNativeArray.ARRAY_HEADER+4L*n;bytes+=size;return FloatArray.fromSegmentShallow(arena.allocate(size,32));}
        IntArray ints(int n) {long size=TornadoNativeArray.ARRAY_HEADER+4L*n;bytes+=size;return IntArray.fromSegmentShallow(arena.allocate(size,32));}
        ByteArray bytes(int n) {long size=TornadoNativeArray.ARRAY_HEADER+(long)n;bytes+=size;return ByteArray.fromSegmentShallow(arena.allocate(size,32));}
        BFloat16Array bfloats(int n) {long size=TornadoNativeArray.ARRAY_HEADER+2L*n;bytes+=size;return BFloat16Array.fromSegmentShallow(arena.allocate(size,32));}
        ByteArray dense(ModelOptMatrix matrix) throws IOException {
            if(!matrix.dtype().equals("BF16")) throw new IOException("Expected BF16 shared/router weights");var out=bytes(Math.toIntExact(matrix.weights().bytes()));
            try(var temporary=Arena.ofConfined();var file=FileChannel.open(matrix.weights().file(),StandardOpenOption.READ)) {
                out.getSegment().copyFrom(file.map(FileChannel.MapMode.READ_ONLY,matrix.weights().offset(),matrix.weights().bytes(),temporary));
            }return out;
        }
        void run(int index) throws IOException {
            try {long time=plan.withGraph(index).execute().getProfilerResult().getDeviceKernelTime();if(time<=0) throw new IOException("MoE graph recorded no CUDA kernel");kernelNanos+=time;}
            catch(RuntimeException error) {throw new IOException("GPU MoE graph "+index+" failed",error);}
        }
        @Override public Result forward(float[] values) throws IOException {
            synchronized(FlashNextMoeExecutor.this) {
                if(released || closed) throw new IOException("GPU MoE layer is closed");if(values.length!=dim) throw new IllegalArgumentException("Wrong MoE input width");
                for(int i=0;i<dim;i++) {if(!Float.isFinite(values[i])) throw new IllegalArgumentException("Nonfinite MoE input");input.set(i,values[i]);}
                run(0);int[] ids=new int[top];float[] routing=new float[top];for(int i=0;i<top;i++) {ids[i]=selected.get(i);routing[i]=routeWeights.get(i);}idDownloads+=8L*top;
                try(var lease=weights.lease()) {
                    for(int i=0;i<top;i++) {
                        var a=lease.get(ups[ids[i]],gates[ids[i]]);var b=lease.get(downs[ids[i]],null);
                        resources1.set(i,a.id());resources2.set(i,b.id());alpha1.set(2*i,a.scale()*inputScale);alpha1.set(2*i+1,0);
                        alpha2.set(2*i,b.scale()*midScale);alpha2.set(2*i+1,0);
                    }
                    peakBytes=Math.max(peakBytes,layerBytes+weights.resident);
                    run(1);run(2);
                }
                hostUploadBytes+=4L*dim+24L*top;hostDownloadBytes+=8L*top+4L*dim;
                float[] result=new float[dim];for(int i=0;i<dim;i++) result[i]=output.get(i);forwards++;
                return new Result(result,ids,routing);
            }
        }
        @Override public void close() throws IOException {
            synchronized(FlashNextMoeExecutor.this) {if(released) return;released=true;try {plan.close();} catch(Exception error) {throw new IOException("Cannot close GPU MoE layer",error);}
                finally {arena.close();if(layers.remove(this)) {layerBytes-=bytes;weights.budget=budget-layerBytes;}}}
        }
        @Override public float[] routingWeights() throws IOException {
            synchronized(FlashNextMoeExecutor.this) {
                if(released || closed) throw new IOException("GPU MoE layer is closed");run(3);
                float[] result=new float[top];for(int i=0;i<top;i++) result[i]=diagnosticWeights.get(i);idDownloads+=4L*result.length;hostDownloadBytes+=4L*result.length;return result;
            }
        }
    }
    public static void pad(FloatArray input,FloatArray output) {
        for(@uk.ac.manchester.tornado.api.annotations.Parallel int i=0;i<output.getSize();i++) output.set(i,i<input.getSize() ? input.get(i) : 0);
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;IOException failure=null;
        for(var layer:List.copyOf(layers)) try {layer.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        try {weights.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}closed=true;if(failure!=null) throw failure;
    }
}

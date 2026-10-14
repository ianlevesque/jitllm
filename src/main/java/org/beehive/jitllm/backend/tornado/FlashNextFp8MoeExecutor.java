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
 * packed native FP8 selected experts and finalization. Only selected IDs and final outputs cross back.
 * Compute plans/workspaces are per layer; expert residency is an independent bounded LRU.
 */
public final class FlashNextFp8MoeExecutor implements MoeExecutor,AutoCloseable {
    public record Statistics(long residentBytes,long layerBytes,long expertBytes,int layers,int registeredExperts,long uploads,long evictions,
                             long expertFileBytes,long forwards,long kernelNanos,long idDownloadBytes,long intermediateDownloadBytes,long peakBytes,long layerBuilds,
                             long expertCacheHits,long expertCacheMisses,long expertEvictedBytes,long hostUploadBytes,long hostDownloadBytes) {}
    private static final int PAD=16;
    private final TornadoDevice device;
    private final long budget;
    private final Set<DeviceLayer> layers=Collections.newSetFromMap(new IdentityHashMap<>());
    private final FlashNextFp8ExpertCache weights;
    private long layerBytes,graphId,forwards,kernelNanos,idDownloads,peakBytes,layerBuilds,traceDownloads,hostUploadBytes,hostDownloadBytes;
    private boolean closed;
    public FlashNextFp8MoeExecutor(int index,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive MoE budget required");this.budget=budget;
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(index);weights=new FlashNextFp8ExpertCache(device,budget);
    }
    public synchronized Statistics statistics() {return new Statistics(layerBytes+weights.resident,layerBytes,weights.resident,layers.size(),weights.cache.size(),weights.uploads,weights.evictions,weights.readBytes,forwards,kernelNanos,idDownloads,traceDownloads,peakBytes,layerBuilds,weights.hits,weights.misses,weights.evictedBytes,hostUploadBytes,hostDownloadBytes);}
    @Override public synchronized Layer newLayer(ModelOptMatrix router,ModelOptMatrix sharedGate,ModelOptMatrix sharedUp,ModelOptMatrix sharedDown,ModelOptMatrix sharedRouter,
                                                ModelOptMatrix[] gates,ModelOptMatrix[] ups,ModelOptMatrix[] downs,int topK,boolean normalize,float inputScale,float intermediateScale) throws IOException {
        if(closed) throw new IllegalStateException("MoE executor is closed");
        if(router.rows()!=512 || gates.length!=512 || ups.length!=512 || downs.length!=512 || topK<=0 || topK>512)
            throw new IllegalArgumentException("Invalid GPU FP8 MoE geometry");
        for(int i=0;i<512;i++) if(!gates[i].dtype().equals("F8_E4M3") || !ups[i].dtype().equals("F8_E4M3") || !downs[i].dtype().equals("F8_E4M3")
                || gates[i].rows()!=gates[0].rows() || gates[i].columns()!=router.columns()
                || gates[i].rows()!=downs[i].columns() || gates[i].columns()!=downs[i].rows() || gates[i].rows()!=ups[i].rows() || gates[i].columns()!=ups[i].columns())
            throw new IOException("Block FP8 up/gate layout requires matching shapes at expert "+i);
        if(sharedGate.columns()!=router.columns() || sharedUp.columns()!=router.columns() || sharedGate.rows()!=sharedUp.rows()
                || sharedDown.columns()!=sharedGate.rows() || sharedDown.rows()!=router.columns()
                || sharedRouter.rows()!=1 || sharedRouter.columns()!=router.columns())
            throw new IOException("Invalid FP8 MoE shared expert shapes");
        var layer=new DeviceLayer(router,sharedGate,sharedUp,sharedDown,sharedRouter,gates,ups,downs,topK,normalize,inputScale,intermediateScale);
        if(layer.bytes>budget-layerBytes-weights.resident) {layer.close();throw new IOException("MoE layer allocation exceeds budget");}
        layers.add(layer);layerBytes+=layer.bytes;weights.budget=budget-layerBytes;layerBuilds++;peakBytes=Math.max(peakBytes,layerBytes+weights.resident);return layer;
    }
    private final class DeviceLayer implements Layer {
        final Arena arena=Arena.ofShared();final ModelOptMatrix[] gates,ups,downs;
        final int dim,hidden,top;
        final FloatArray input,output,routeWeights,diagnosticWeights,scale1,scale2;
        final IntArray selected,resources1,resources2;
        final TornadoExecutionPlan plan;
        final FloatArray traceFc1,traceMid,traceDown,traceShared,traceRouter,traceScales;
        long bytes;boolean released;
        DeviceLayer(ModelOptMatrix router,ModelOptMatrix sg,ModelOptMatrix su,ModelOptMatrix sd,ModelOptMatrix sr,
                    ModelOptMatrix[] gates,ModelOptMatrix[] ups,ModelOptMatrix[] downs,int top,boolean normalize,float inputScale,float midScale) throws IOException {
            this.gates=gates.clone();this.ups=ups.clone();this.downs=downs.clone();this.top=top;
            dim=router.columns();hidden=gates[0].rows();TornadoExecutionPlan built=null;
            try {
                if(dim%128!=0 || hidden%128!=0) throw new IOException("Unsupported packed MoE dimensions");
                var rw=dense(router);var gw=dense(sg);var uw=dense(su);var dw=dense(sd);var sw=dense(sr);
                input=floats(dim);output=floats(dim);selected=ints(top);routeWeights=floats(top);diagnosticWeights=floats(top);
                var logits=floats(512);var sharedGate=floats(sg.rows());var sharedUp=floats(su.rows());var sharedMid=floats(sg.rows());var scratch=floats(1);
                var shared=floats(dim);var sharedRouter=floats(1);
                var sharedInput=bfloats(dim);var sharedGateBf=bfloats(sg.rows());var sharedUpBf=bfloats(su.rows());var sharedMidBf=bfloats(sg.rows());var sharedBf=bfloats(dim);
                var padded=floats(PAD*dim);var packed=bytes(PAD*dim);var sf=floats(PAD*dim/128);
                var mid=floats(top*PAD*hidden);var midPacked=bytes(top*PAD*hidden);var midSf=floats(top*PAD*hidden/128);
                var partial1=floats(top*PAD*2*hidden*(dim/128));var partial2=floats(top*PAD*dim*(hidden/128));
                var upGate=bfloats(top*PAD*2*hidden);var down=bfloats(top*PAD*dim);
                var table=floats(127);for(int i=0;i<127;i++) table.set(i,FloatFormats.e4m3(i));
                var control=floats(3);control.set(0,1);control.set(2,1);
                resources1=ints(top);resources2=ints(top);scale1=floats(top*(2*hidden/128)*(dim/128));scale2=floats(top*(dim/128)*(hidden/128));
                String id="flashnextFp8Moe"+(graphId++);
                var preparation=new TaskGraph(id)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,rw,gw,uw,dw,sw,padded,table,control,sharedInput,sharedGateBf,sharedUpBf,sharedMidBf,sharedBf)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,input)
                        .task("router",FlashNextWeightKernels::denseMatvec,input,rw,logits,dim,0).task("routerBoundary",FlashNextMoeKernels::bf16Boundary,logits)
                        .task("packShared",FlashNextWeightKernels::bf16GemmInput,input,sharedInput,dim,0)
                        .libraryTask("gate",FlashNextSharedBf16Library::project,sg.rows(),dim,gw,sharedInput,sharedGateBf)
                        .task("gateBoundary",FlashNextWeightKernels::bf16GemmOutput,sharedGateBf,sharedGate)
                        .libraryTask("up",FlashNextSharedBf16Library::project,su.rows(),dim,uw,sharedInput,sharedUpBf)
                        .task("upBoundary",FlashNextWeightKernels::bf16GemmOutput,sharedUpBf,sharedUp)
                        .task("sharedActivation",FlashNextMoeKernels::swiglu,sharedGate,sharedUp,sharedMid)
                        .task("packSharedMid",FlashNextWeightKernels::bf16GemmInput,sharedMid,sharedMidBf,sg.rows(),0)
                        .libraryTask("sharedDown",FlashNextSharedBf16Library::project,dim,sg.rows(),dw,sharedMidBf,sharedBf)
                        .task("sharedBoundary",FlashNextWeightKernels::bf16GemmOutput,sharedBf,shared)
                        .task("sharedRouter",FlashNextWeightKernels::denseMatvec,input,sw,sharedRouter,dim,0).task("sharedRouterBoundary",FlashNextMoeKernels::bf16Boundary,sharedRouter)
                        .task("route",FlashNextMoeKernels::route,new KernelContext(),logits,selected,routeWeights,normalize ? 1 : 0)
                        .task("pad",FlashNextFp8MoeExecutor::pad,input,padded)
                        .task("seedEncoding",FlashNextFp8Kernels::seed,table,control,scratch)
                        .persistOnDevice(padded,routeWeights,shared,sharedRouter,table,control)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,selected,routeWeights);
                var fc1=new TaskGraph(id+"Fc1").consumeFromDevice(id,padded,table,control,routeWeights)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,packed,sf,upGate,partial1)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,resources1,scale1)
                        .task("encode",FlashNextFp8Kernels::quantize,padded,table,packed,sf)
                        .libraryTask("experts",FlashNextFp8Library::experts,2*hidden,PAD,dim,top,resources1,packed,control,partial1,0)
                        .task("reduce",FlashNextFp8Kernels::reduceExperts,partial1,scale1,sf,routeWeights,upGate,2*hidden,dim,PAD,0,0)
                        .task("activation",FlashNextMoeKernels::expertActivation,upGate,mid,hidden,PAD).persistOnDevice(mid,upGate);
                var fc2=new TaskGraph(id+"Fc2").consumeFromDevice(id,table,control,routeWeights,shared,sharedRouter).consumeFromDevice(id+"Fc1",mid)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,midPacked,midSf,down,partial2)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,resources2,scale2)
                        .task("encode",FlashNextFp8Kernels::quantize,mid,table,midPacked,midSf)
                        .libraryTask("experts",FlashNextFp8Library::experts,dim,PAD,hidden,top,resources2,midPacked,control,partial2,PAD*hidden)
                        .task("reduce",FlashNextFp8Kernels::reduceExperts,partial2,scale2,midSf,routeWeights,down,dim,hidden,PAD,1,1)
                        .task("finish",FlashNextFp8Kernels::finish,down,shared,sharedRouter,output,top,PAD)
                        .persistOnDevice(down,midSf).transferToHost(DataTransferMode.EVERY_EXECUTION,output);
                var diagnostic=new TaskGraph(id+"Weights").consumeFromDevice(id,routeWeights)
                        .task("read",FlashNextMoeKernels::readWeights,routeWeights,diagnosticWeights).transferToHost(DataTransferMode.EVERY_EXECUTION,diagnosticWeights);
                traceFc1=floats(top*2*hidden);traceMid=floats(top*hidden);traceDown=floats(top*dim);traceShared=floats(dim);traceRouter=floats(1);traceScales=floats(top*hidden/128);
                var trace=new TaskGraph(id+"Trace").consumeFromDevice(id,shared,sharedRouter).consumeFromDevice(id+"Fc1",upGate,mid).consumeFromDevice(id+"Fc2",down,midSf)
                        .task("fc1",FlashNextFp8Kernels::readExpertBf16,upGate,traceFc1,2*hidden,PAD)
                        .task("mid",FlashNextFp8Kernels::readExpertFloat,mid,traceMid,hidden,PAD)
                        .task("down",FlashNextFp8Kernels::readExpertBf16,down,traceDown,dim,PAD)
                        .task("shared",FlashNextMoeKernels::readWeights,shared,traceShared).task("sharedRouter",FlashNextMoeKernels::readWeights,sharedRouter,traceRouter)
                        .task("scales",FlashNextFp8Kernels::readExpertFloat,midSf,traceScales,hidden/128,PAD)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,traceFc1,traceMid,traceDown,traceShared,traceRouter,traceScales);
                var worker=new WorkerGrid1D(512);worker.setLocalWork(512,1,1);var scheduler=new GridScheduler(id+".route",worker);
                bytes+=FlashNextFp8Library.WORKSPACE_BYTES+FlashNextSharedBf16Library.WORKSPACE_BYTES;
                built=new TornadoExecutionPlan(preparation.snapshot(),fc1.snapshot(),fc2.snapshot(),diagnostic.snapshot(),trace.snapshot()).withDevice(device).withGridScheduler(scheduler)
                        .withProfiler(ProfilerMode.SILENT).withCompilerFlags(TornadoVMBackendType.CUDA,"--use_fast_math");plan=built;
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
            synchronized(FlashNextFp8MoeExecutor.this) {
                if(released || closed) throw new IOException("GPU MoE layer is closed");if(values.length!=dim) throw new IllegalArgumentException("Wrong MoE input width");
                for(int i=0;i<dim;i++) {if(!Float.isFinite(values[i])) throw new IllegalArgumentException("Nonfinite MoE input");input.set(i,values[i]);}
                run(0);int[] ids=new int[top];float[] routing=new float[top];for(int i=0;i<top;i++) {ids[i]=selected.get(i);routing[i]=routeWeights.get(i);}idDownloads+=8L*top;
                try(var lease=weights.lease()) {
                    for(int i=0;i<top;i++) {
                        var a=lease.get(ups[ids[i]],gates[ids[i]]);var b=lease.get(downs[ids[i]],null);
                        resources1.set(i,a.id());resources2.set(i,b.id());
                        for(int j=0;j<a.scales().length;j++) scale1.set(i*a.scales().length+j,a.scales()[j]);
                        for(int j=0;j<b.scales().length;j++) scale2.set(i*b.scales().length+j,b.scales()[j]);
                    }
                    peakBytes=Math.max(peakBytes,layerBytes+weights.resident);
                    run(1);run(2);
                }
                hostUploadBytes+=4L*dim+4L*(resources1.getSize()+scale1.getSize()+resources2.getSize()+scale2.getSize());hostDownloadBytes+=8L*top+4L*dim;
                float[] result=new float[dim];for(int i=0;i<dim;i++) result[i]=output.get(i);forwards++;
                return new Result(result,ids,routing);
            }
        }
        @Override public Map<String,float[]> intermediates() throws IOException {
            synchronized(FlashNextFp8MoeExecutor.this) {
                if(released || closed) throw new IOException("GPU MoE layer is closed");
                run(4);var values=new LinkedHashMap<String,float[]>();
                String[] names={"up_gate","mid","down","shared","shared_router","mid_scales"};FloatArray[] arrays={traceFc1,traceMid,traceDown,traceShared,traceRouter,traceScales};
                for(int i=0;i<names.length;i++) {float[] row=new float[arrays[i].getSize()];for(int j=0;j<row.length;j++) row[j]=arrays[i].get(j);values.put(names[i],row);traceDownloads+=4L*row.length;}
                return values;
            }
        }
        @Override public void close() throws IOException {
            synchronized(FlashNextFp8MoeExecutor.this) {if(released) return;released=true;try {plan.close();} catch(Exception error) {throw new IOException("Cannot close GPU MoE layer",error);}
                finally {arena.close();if(layers.remove(this)) {layerBytes-=bytes;weights.budget=budget-layerBytes;}}}
        }
        @Override public float[] routingWeights() throws IOException {
            synchronized(FlashNextFp8MoeExecutor.this) {
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

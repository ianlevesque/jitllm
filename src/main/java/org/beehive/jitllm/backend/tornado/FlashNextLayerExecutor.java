package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor.GroupedShape;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor.PleShape;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextLayerKernels;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextDeltaNetKernels;
import org.beehive.jitllm.format.safetensors.FloatFormats;
import org.beehive.jitllm.format.safetensors.ModelOptMatrix;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** CUDA HC arithmetic and PLE state. Only gathered rows and projection outputs cross from the host;
 * no PLE table is allocated or mapped on the GPU. Logical buffers and checkpoint banks are bounded.
 */
public final class FlashNextLayerExecutor implements org.beehive.jitllm.inference.op.FlashNextLayerExecutor,AutoCloseable {
    public record Statistics(long residentBytes,int hyperConnections,int pleSessions,int liveSnapshots,
                             long hyperMixes,long hyperCombines,long pleForwards,long checkpoints,long restores,
                             long resets,long kernelNanos,long stateHostDownloadBytes,long hostUploadBytes,long hostDownloadBytes) {}
    private final TornadoDevice device;
    private final int capacity;
    private final boolean nativeGroupedNormalization;
    private final long budget;
    private final Set<Owned> resources=Collections.newSetFromMap(new IdentityHashMap<>());
    private long resident,graphId,hyperMixes,hyperCombines,pleForwards,checkpoints,restores,resets,kernelNanos,hostUploadBytes,hostDownloadBytes;
    private boolean closed;

    public FlashNextLayerExecutor(int deviceIndex,int checkpointCapacity,long budget) {
        this(deviceIndex,checkpointCapacity,budget,false);
    }
    public FlashNextLayerExecutor(int deviceIndex,int checkpointCapacity,long budget,boolean nativeGroupedNormalization) {
        this.nativeGroupedNormalization=nativeGroupedNormalization;
        if(checkpointCapacity<1 || budget<=0) throw new IllegalArgumentException("Positive layer budget and checkpoint capacity required");
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);
        capacity=checkpointCapacity;this.budget=budget;
    }
    public synchronized Statistics statistics() {
        int hyper=0,ple=0,active=0;
        for(var resource:resources) {
            if(resource instanceof Hyper) hyper++;
            if(resource instanceof Ple session) {ple++;for(boolean used:session.occupied) if(used) active++;}
        }
        return new Statistics(resident,hyper,ple,active,hyperMixes,hyperCombines,pleForwards,checkpoints,restores,resets,kernelNanos,0,hostUploadBytes,hostDownloadBytes);
    }
    private void requireOpen() {if(closed) throw new IllegalStateException("GPU layer executor is closed");}
    private <T extends Owned> T register(T resource) {resources.add(resource);resident+=resource.bytes;return resource;}

    private abstract class Owned implements AutoCloseable {
        final Arena arena=Arena.ofShared();
        final List<TornadoExecutionPlan> plans=new ArrayList<>();
        long bytes;
        boolean released;
        void reserve(long size) {
            bytes=Math.addExact(bytes,size);
            if(bytes>budget-resident) throw new IllegalStateException("GPU HC/PLE buffers exceed budget "+budget);
        }
        FloatArray array(int size) {
            long length=Math.addExact(TornadoNativeArray.ARRAY_HEADER,Math.multiplyExact(4L,size));reserve(length);
            return FloatArray.fromSegmentShallow(arena.allocate(length,32));
        }
        FloatArray array(float[] values) {var buffer=array(values.length);put(buffer,values);return buffer;}
        BFloat16Array bfloats(int size) {
            long length=Math.addExact(TornadoNativeArray.ARRAY_HEADER,Math.multiplyExact(2L,size));reserve(length);
            return BFloat16Array.fromSegmentShallow(arena.allocate(length,32));
        }
        IntArray control(int size) {
            long length=TornadoNativeArray.ARRAY_HEADER+4L*size;reserve(length);
            return IntArray.fromSegmentShallow(arena.allocate(length,32));
        }
        void check() {requireOpen();if(released) throw new IllegalStateException("GPU layer resource is closed");}
        TornadoExecutionPlan plan(TaskGraph graph) {
            var plan=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT);
            plans.add(plan);return plan;
        }
        TornadoExecutionResult run(TornadoExecutionPlan plan) throws IOException {
            check();
            try {
                var result=plan.execute();long nanos=result.getProfilerResult().getDeviceKernelTime();
                if(nanos<=0) throw new IOException("GPU HC/PLE recorded no kernel execution");
                kernelNanos+=nanos;return result;
            } catch(RuntimeException failure) {throw new IOException("GPU HC/PLE execution failed",failure);}
        }
        void failed(Throwable failure) {try {close();} catch(IOException closing) {failure.addSuppressed(closing);}}
        void onClosing() {}
        @Override public void close() throws IOException {
            synchronized(FlashNextLayerExecutor.this) {
                if(released) return;released=true;onClosing();IOException failure=null;
                for(var plan:plans) try {plan.close();} catch(Exception closing) {
                    if(failure==null) failure=new IOException("Cannot release GPU HC/PLE plan",closing);else failure.addSuppressed(closing);
                }
                arena.close();if(resources.remove(this)) resident-=bytes;
                if(failure!=null) throw failure;
            }
        }
    }
    private static void put(FloatArray buffer,float[] values) {
        if(values.length!=buffer.getSize()) throw new IllegalArgumentException("Wrong GPU layer input width");
        for(int i=0;i<values.length;i++) buffer.set(i,values[i]);
    }
    private static float[] read(FloatArray buffer) {float[] values=new float[buffer.getSize()];for(int i=0;i<values.length;i++) values[i]=buffer.get(i);return values;}

    @Override public synchronized HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization) {
        requireOpen();
        if(lowrank<=0 || normalization.length!=shape.width()) throw new IllegalArgumentException("Wrong GPU HC shape/weights");
        try {return register(new Hyper(shape,lowrank,normalization,null,null,null));}
        catch(IOException error) {throw new java.io.UncheckedIOException(error);}
    }
    @Override public synchronized HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization,float[] injectionWeights) {
        requireOpen();
        if(lowrank<=0 || normalization.length!=shape.width() || (injectionWeights!=null && injectionWeights.length!=Math.multiplyExact(shape.branches(),shape.width())))
            throw new IllegalArgumentException("Wrong GPU HC shape/weights");
        try {return register(new Hyper(shape,lowrank,normalization,injectionWeights,null,null));}
        catch(IOException error) {throw new java.io.UncheckedIOException(error);}
    }
    @Override public synchronized HyperConnection newHyperConnection(GroupedShape shape,int lowrank,float[] normalization,float[] injectionWeights,ModelOptMatrix down,ModelOptMatrix up) throws IOException {
        requireOpen();
        if(lowrank<=0 || normalization.length!=shape.width() || (injectionWeights!=null && injectionWeights.length!=Math.multiplyExact(shape.branches(),shape.width())))
            throw new IllegalArgumentException("Wrong GPU HC shape/weights");
        boolean nativeMix=nativeGroupedNormalization && shape.branches()==4 && shape.dimension()==2560 && lowrank==320
                && !Boolean.getBoolean("jitllm.qwen4.legacyHcMix");
        return register(new Hyper(shape,lowrank,normalization,injectionWeights,nativeMix ? down : null,nativeMix ? up : null));
    }
    private final class Hyper extends Owned implements HyperConnection {
        final FloatArray normInput,normOutput,low,mixInput,gates,mixed,block,residual,injection,combined;
        final IntArray lowFused,mixFused;
        final TornadoExecutionPlan normalizePlan,lowPlan,mixPlan,combinePlan,nativeCombinePlan,nativeMixPlan;
        final FloatArray nativeRaw;
        Hyper(GroupedShape shape,int lowrank,float[] normalization,float[] injectionWeights,ModelOptMatrix down,ModelOptMatrix up) throws IOException {
            try {
                normInput=array(shape.width());normOutput=array(shape.width());var weights=array(normalization);
                low=array(lowrank);lowFused=control(1);
                mixInput=array(shape.width());gates=array(shape.width());mixed=array(shape.dimension());mixFused=control(1);
                block=array(shape.dimension());residual=array(shape.width());injection=array(shape.branches());combined=array(shape.width());
                String id="flashnextHyper"+(graphId++);
                if(down!=null && up!=null) {
                    if(!down.dtype().equals("BF16") || !up.dtype().equals("BF16") || down.rows()!=320 || down.columns()!=10240
                            || up.rows()!=10240 || up.columns()!=320) throw new IOException("Native HC mix requires the qualified BF16 matrices");
                    var packed=bfloats(shape.width());var dw=matrix(down);var uw=matrix(up);
                    nativeRaw=array(16*lowrank);var out=bfloats(shape.dimension());var counters=control(3);
                    nativeMixPlan=plan(new TaskGraph(id+"NativeMix")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION,packed,dw,uw,nativeRaw,out,counters)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION,mixInput)
                            .task("pack",org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels::bf16GemmInput,mixInput,packed,shape.width(),1)
                            .libraryTask("mix",FlashNextHcMixLibrary::mix,packed,dw,uw,nativeRaw,out,counters)
                            .task("read",org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels::bf16GemmOutput,out,mixed)
                            .transferToHost(DataTransferMode.UNDER_DEMAND,mixed,nativeRaw));
                } else {nativeMixPlan=null;nativeRaw=null;}
                var normGraph=new TaskGraph(id+"Norm").transferToDevice(DataTransferMode.FIRST_EXECUTION,weights)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,normInput);
                if(nativeGroupedNormalization && shape.dimension()%512==0 && shape.dimension()<=16384) {
                    normGraph.libraryTask("normalize",FlashNextRmsNormLibrary::grouped,shape.branches(),shape.dimension(),shape.epsilon(),normInput,weights,normOutput)
                            .task("boundary",org.beehive.jitllm.backend.tornado.kernels.FlashNextMoeKernels::bf16Boundary,normOutput);
                } else normGraph.task("normalize",FlashNextLayerKernels::groupedNorm,normInput,weights,normOutput,shape.dimension(),shape.epsilon());
                normalizePlan=plan(normGraph.transferToHost(DataTransferMode.UNDER_DEMAND,normOutput));
                var lowGraph=new TaskGraph(id+"Low").transferToDevice(DataTransferMode.EVERY_EXECUTION,low,lowFused);
                if(nativeGroupedNormalization && shape.branches()==4) lowGraph.libraryTask("activate",FlashNextRmsNormLibrary::activate,shape.branches(),lowrank,low,lowFused);
                else lowGraph.task("activate",FlashNextLayerKernels::activateLow,low,lowFused,shape.branches());
                lowPlan=plan(lowGraph.transferToHost(DataTransferMode.UNDER_DEMAND,low));
                var mixGraph=new TaskGraph(id+"Mix").transferToDevice(DataTransferMode.EVERY_EXECUTION,mixInput,gates,mixFused);
                if(nativeGroupedNormalization && shape.branches()==4) mixGraph.libraryTask("mix",FlashNextRmsNormLibrary::mix,shape.branches(),shape.dimension(),mixInput,gates,mixFused,mixed);
                else mixGraph.task("mix",FlashNextLayerKernels::mix,mixInput,gates,mixFused,mixed,shape.branches());
                mixPlan=plan(mixGraph.transferToHost(DataTransferMode.UNDER_DEMAND,mixed));
                combinePlan=plan(new TaskGraph(id+"Combine").transferToDevice(DataTransferMode.EVERY_EXECUTION,block,residual,injection)
                        .task("combine",FlashNextLayerKernels::combine,block,residual,injection,combined,shape.branches())
                        .transferToHost(DataTransferMode.UNDER_DEMAND,combined));
                if(injectionWeights!=null && shape.dimension()%8==0 && shape.width()%2048==0) {
                    var injectWeights=array(injectionWeights);
                    nativeCombinePlan=plan(new TaskGraph(id+"NativeCombine").transferToDevice(DataTransferMode.FIRST_EXECUTION,injectWeights)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION,block,residual,mixInput)
                            .libraryTask("combine",FlashNextRmsNormLibrary::combine,shape.branches(),shape.dimension(),block,residual,mixInput,injectWeights,combined)
                            .task("boundary",org.beehive.jitllm.backend.tornado.kernels.FlashNextMoeKernels::bf16Boundary,combined)
                            .transferToHost(DataTransferMode.UNDER_DEMAND,combined));
                } else nativeCombinePlan=null;
            } catch(IOException|RuntimeException|Error failure) {failed(failure);throw failure;}
        }
        private BFloat16Array matrix(ModelOptMatrix matrix) throws IOException {
            var result=bfloats(Math.multiplyExact(matrix.rows(),matrix.columns()));
            try(var temporary=Arena.ofConfined();var file=FileChannel.open(matrix.weights().file(),StandardOpenOption.READ)) {
                result.getSegment().copyFrom(file.map(FileChannel.MapMode.READ_ONLY,matrix.weights().offset(),matrix.weights().bytes(),temporary));
            }
            return result;
        }
        @Override public boolean projectsMix() {return nativeMixPlan!=null;}
        @Override public float[] mixNormalized(float[] normalized,java.util.function.BiConsumer<String,float[]> trace) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {
                check();if(nativeMixPlan==null) throw new UnsupportedOperationException("Native HC mix is unavailable");
                put(mixInput,normalized);var result=run(nativeMixPlan);result.transferToHost(mixed);
                hostUploadBytes+=4L*mixInput.getSize();hostDownloadBytes+=4L*mixed.getSize();
                if(trace!=null) {result.transferToHost(nativeRaw);hostDownloadBytes+=4L*nativeRaw.getSize();float[] raw=new float[320];for(int i=0;i<raw.length;i++) raw[i]=nativeRaw.get(i);trace.accept("down_raw",raw);}
                hyperMixes++;return read(mixed);
            }
        }
        @Override public float[] normalize(float[] hidden) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {check();put(normInput,hidden);run(normalizePlan).transferToHost(normOutput);
                hostUploadBytes+=4L*normInput.getSize();hostDownloadBytes+=4L*normOutput.getSize();return read(normOutput);}
        }
        @Override public float[] activateLow(float[] projected,boolean fused) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {check();put(low,projected);lowFused.set(0,fused ? 1 : 0);run(lowPlan).transferToHost(low);
                hostUploadBytes+=4L*low.getSize()+4;hostDownloadBytes+=4L*low.getSize();return read(low);}
        }
        @Override public float[] mix(float[] normalized,float[] projectedGates,boolean fused) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {check();put(mixInput,normalized);put(gates,projectedGates);mixFused.set(0,fused ? 1 : 0);
                run(mixPlan).transferToHost(mixed);hostUploadBytes+=4L*(mixInput.getSize()+gates.getSize())+4;hostDownloadBytes+=4L*mixed.getSize();hyperMixes++;return read(mixed);}
        }
        @Override public float[] combine(float[] blockValues,float[] residualValues,float[] projectedInjection) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {check();put(block,blockValues);put(residual,residualValues);put(injection,projectedInjection);
                run(combinePlan).transferToHost(combined);hostUploadBytes+=4L*(block.getSize()+residual.getSize()+injection.getSize());hostDownloadBytes+=4L*combined.getSize();hyperCombines++;return read(combined);}
        }
        @Override public boolean projectsInjection() {return nativeCombinePlan!=null;}
        @Override public float[] combineNormalized(float[] blockValues,float[] residualValues,float[] normalized) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {check();if(nativeCombinePlan==null) throw new UnsupportedOperationException("Native HC injection is unavailable");
                put(block,blockValues);put(residual,residualValues);put(mixInput,normalized);
                run(nativeCombinePlan).transferToHost(combined);hostUploadBytes+=4L*(block.getSize()+residual.getSize()+mixInput.getSize());hostDownloadBytes+=4L*combined.getSize();hyperCombines++;return read(combined);}
        }
    }

    @Override public synchronized PleSession newPleSession(PleShape shape,float[] keyNorm,float[] queryNorm,float[] convolutionNorm,float[] convolution) {
        requireOpen();int width=shape.grouped().width();
        if(keyNorm.length!=width || queryNorm.length!=width || convolutionNorm.length!=width || convolution.length!=Math.multiplyExact(width,shape.kernel()))
            throw new IllegalArgumentException("Wrong GPU PLE shape/weights");
        return register(new Ple(shape,keyNorm,queryNorm,convolutionNorm,convolution));
    }
    private final class Ple extends Owned implements PleSession {
        final FloatArray state,bank,hidden,key,value,output;
        final IntArray control;
        final TornadoExecutionPlan execution;
        final boolean[] occupied=new boolean[capacity];
        long generation;
        Ple(PleShape shape,float[] keyNorm,float[] queryNorm,float[] convolutionNorm,float[] convolution) {
            try {
                var grouped=shape.grouped();int width=grouped.width();
                state=array(shape.stateSize());bank=array(Math.multiplyExact(shape.stateSize(),capacity));
                hidden=array(width);key=array(width);value=array(grouped.dimension());output=array(width);control=control(4);
                FloatArray keyWeights=array(keyNorm),queryWeights=array(queryNorm),convNorm=array(convolutionNorm),conv=array(convolution);
                FloatArray normalizedKey=array(width),normalizedQuery=array(width),gated=array(width),normalized=array(width);
                var graph=new TaskGraph("flashnextPle"+(graphId++))
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,state,keyWeights,queryWeights,convNorm,conv)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,control,hidden,key,value);
                if(shape.stateSize()>0) graph.task("checkpoint",FlashNextDeltaNetKernels::checkpoint,control,state,bank);
                graph.task("normalize",FlashNextLayerKernels::normalizePle,control,key,hidden,keyWeights,queryWeights,normalizedKey,normalizedQuery,grouped.dimension(),grouped.epsilon())
                        .task("gate",FlashNextLayerKernels::pleGate,control,normalizedKey,normalizedQuery,value,gated,grouped.dimension(),FloatFormats.roundBf16(1e-6f))
                        .task("convNorm",FlashNextLayerKernels::normalizePleConvolution,control,gated,convNorm,normalized,grouped.dimension(),grouped.epsilon())
                        .task("convolve",FlashNextLayerKernels::pleConvolution,control,normalized,gated,hidden,conv,state,output,shape.kernel(),shape.dilation())
                        .transferToHost(DataTransferMode.UNDER_DEMAND,output);
                execution=plan(graph);
            } catch(RuntimeException|Error failure) {failed(failure);throw failure;}
        }
        TornadoExecutionResult execute(int action,int slot) throws IOException {check();control.set(0,action);control.set(1,slot);
            var result=run(execution);hostUploadBytes+=16+4L*(hidden.getSize()+key.getSize()+value.getSize());return result;}
        @Override public float[] forward(float[] residual,float[] projectedKey,float[] projectedValue,boolean fusedDecode,boolean addResidual) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {
                check();put(hidden,residual);put(key,projectedKey);put(value,projectedValue);
                control.set(2,addResidual ? 1 : 0);control.set(3,fusedDecode ? 1 : 0);
                execute(0,0).transferToHost(output);hostDownloadBytes+=4L*output.getSize();pleForwards++;return read(output);
            }
        }
        private final class Saved implements Snapshot {
            final Ple owner=Ple.this;
            final int slot;
            final long epoch=generation;
            boolean disposed;
            Saved(int slot) {this.slot=slot;}
            @Override public void close() {
                synchronized(FlashNextLayerExecutor.this) {if(!disposed && epoch==generation) occupied[slot]=false;disposed=true;}
            }
        }
        @Override public Snapshot snapshot() throws IOException {
            synchronized(FlashNextLayerExecutor.this) {
                check();int slot=0;while(slot<capacity && occupied[slot]) slot++;
                if(slot==capacity) throw new IllegalStateException("GPU PLE checkpoint pool exhausted: "+capacity);
                execute(1,slot);occupied[slot]=true;checkpoints++;return new Saved(slot);
            }
        }
        @Override public void restore(Snapshot snapshot) throws IOException {
            synchronized(FlashNextLayerExecutor.this) {
                check();
                if(!(snapshot instanceof Saved saved) || saved.owner!=this || saved.disposed || saved.epoch!=generation)
                    throw new IllegalArgumentException("Invalid GPU PLE checkpoint owner/lifetime");
                execute(2,saved.slot);restores++;
            }
        }
        @Override public void reset() throws IOException {
            synchronized(FlashNextLayerExecutor.this) {execute(3,0);generation++;Arrays.fill(occupied,false);resets++;}
        }
        @Override void onClosing() {generation++;Arrays.fill(occupied,false);}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;IOException failure=null;
        for(var resource:List.copyOf(resources)) try {resource.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        closed=true;if(failure!=null) throw failure;
    }
}

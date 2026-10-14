package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.*;
import org.beehive.jitllm.inference.op.DeltaNetExecutor;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextDeltaNetKernels;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Request-owned GPU recurrence with a bounded, reusable pool of device checkpoints.
 * The live recurrent state is uploaded once; forward, save, restore and reset keep it on device.
 */
public final class FlashNextDeltaNetExecutor implements DeltaNetExecutor,AutoCloseable {
    public record Statistics(long residentBytes,int sessions,int liveSnapshots,long forwards,long checkpoints,
                             long restores,long resets,long kernelNanos,long stateHostDownloadBytes,long hostUploadBytes,long hostDownloadBytes) {}
    private final TornadoDevice device;
    private final int capacity;
    private final long budget;
    private final Set<DeviceSession> sessions=Collections.newSetFromMap(new IdentityHashMap<>());
    private long resident,graphId,forwards,checkpoints,restores,resets,kernelNanos,hostUploadBytes,hostDownloadBytes;
    private boolean closed;

    public FlashNextDeltaNetExecutor(int deviceIndex,int checkpointCapacity,long budget) {
        if(checkpointCapacity<1 || budget<=0) throw new IllegalArgumentException("Positive checkpoint capacity and budget required");
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);
        capacity=checkpointCapacity;this.budget=budget;
    }
    public synchronized Statistics statistics() {
        int active=0;for(var session:sessions) for(boolean used:session.occupied) if(used) active++;
        return new Statistics(resident,sessions.size(),active,forwards,checkpoints,restores,resets,kernelNanos,0,hostUploadBytes,hostDownloadBytes);
    }
    private long footprint(Shape shape) {
        long floats=(long)shape.stateSize()*(capacity+1)+shape.channels()*2L+shape.channels()*shape.convolutionKernel()
                +shape.keyHeads()*shape.keyDim()*2L+shape.valueHeads()*shape.valueDim()*3L
                +shape.valueHeads()*4L+shape.valueDim();
        return Math.addExact(Math.multiplyExact(floats,4),16L*TornadoNativeArray.ARRAY_HEADER+8);
    }
    @Override public synchronized Session newSession(Shape shape,float[] convolution,float[] normalization,float[] aLog,float[] dtBias) {
        if(closed) throw new IllegalStateException("GPU DeltaNet executor is closed");
        if(convolution.length!=shape.channels()*shape.convolutionKernel() || normalization.length!=shape.valueDim()
                || aLog.length!=shape.valueHeads() || dtBias.length!=shape.valueHeads())
            throw new IllegalArgumentException("Invalid DeltaNet weights");
        long bytes=footprint(shape);
        if(bytes>budget-resident) throw new IllegalStateException("GPU DeltaNet state exceeds budget "+budget);
        var session=new DeviceSession(shape,convolution,normalization,aLog,dtBias,bytes);
        sessions.add(session);resident+=bytes;return session;
    }
    private final class DeviceSession implements Session {
        final Shape shape;
        final long bytes;
        final Arena arena=Arena.ofShared();
        final FloatArray state,bank,projected,gate,alpha,beta,output;
        final IntArray control;
        final TornadoExecutionPlan plan;
        final boolean[] occupied=new boolean[capacity];
        long generation;
        boolean released;
        DeviceSession(Shape shape,float[] conv,float[] norm,float[] aLog,float[] dtBias,long bytes) {
            this.shape=shape;this.bytes=bytes;
            try {
                state=array(shape.stateSize());bank=array(Math.multiplyExact(shape.stateSize(),capacity));
                projected=array(shape.channels());gate=array(shape.valueHeads()*shape.valueDim());
                alpha=array(shape.valueHeads());beta=array(shape.valueHeads());output=array(gate.getSize());
                control=IntArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+8,32));
                FloatArray weights=array(conv),normalization=array(norm),log=array(aLog),bias=array(dtBias);
                FloatArray convolved=array(shape.channels()),query=array(shape.keyHeads()*shape.keyDim()),key=array(query.getSize()),core=array(gate.getSize());
                var graph=new TaskGraph("flashnextDelta"+(graphId++))
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,state,weights,normalization,log,bias)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,control,projected,gate,alpha,beta)
                        .task("checkpoint",FlashNextDeltaNetKernels::checkpoint,control,state,bank)
                        .task("convolve",FlashNextDeltaNetKernels::convolve,control,projected,weights,state,convolved,shape.convolutionKernel())
                        .task("normalize",FlashNextDeltaNetKernels::normalizeQk,control,convolved,query,key,shape.keyDim())
                        .task("recur",FlashNextDeltaNetKernels::recur,control,convolved,query,key,alpha,beta,log,bias,state,core,shape.keyDim(),shape.valueDim(),shape.convolutionSize())
                        .task("gate",FlashNextDeltaNetKernels::gateNorm,control,core,gate,normalization,output,shape.valueDim(),shape.epsilon())
                        .transferToHost(DataTransferMode.UNDER_DEMAND,output);
                plan=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT);
            } catch(RuntimeException|Error failure) {arena.close();throw failure;}
        }
        FloatArray array(int size) {return FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*size,32));}
        FloatArray array(float[] values) {var array=array(values.length);for(int i=0;i<values.length;i++) array.set(i,values[i]);return array;}
        void requireOpen() {if(released || closed) throw new IllegalStateException("GPU DeltaNet session is closed");}
        TornadoExecutionResult execute(int action,int slot) throws IOException {
            requireOpen();control.set(0,action);control.set(1,slot);
            try {
                var result=plan.execute();long nanos=result.getProfilerResult().getDeviceKernelTime();
                if(nanos<=0) throw new IOException("GPU DeltaNet recorded no kernel execution");
                kernelNanos+=nanos;hostUploadBytes+=8+4L*(projected.getSize()+gate.getSize()+alpha.getSize()+beta.getSize());return result;
            } catch(RuntimeException failure) {throw new IOException("GPU DeltaNet execution failed",failure);}
        }
        @Override public float[] forward(float[] qkv,float[] z,float[] a,float[] b) throws IOException {
            synchronized(FlashNextDeltaNetExecutor.this) {
                requireOpen();
                if(qkv.length!=projected.getSize() || z.length!=gate.getSize() || a.length!=alpha.getSize() || b.length!=beta.getSize())
                    throw new IllegalArgumentException("Invalid DeltaNet projection shape");
                put(projected,qkv);put(gate,z);put(alpha,a);put(beta,b);
                execute(0,0).transferToHost(output);hostDownloadBytes+=4L*output.getSize();forwards++;
                float[] values=new float[output.getSize()];for(int i=0;i<values.length;i++) values[i]=output.get(i);
                return values;
            }
        }
        void put(FloatArray target,float[] values) {for(int i=0;i<values.length;i++) target.set(i,values[i]);}
        final class DeviceSnapshot implements Snapshot {
            final DeviceSession owner=DeviceSession.this;
            final int slot;
            final long epoch=generation;
            boolean disposed;
            DeviceSnapshot(int slot) {this.slot=slot;}
            @Override public void close() {
                synchronized(FlashNextDeltaNetExecutor.this) {
                    if(!disposed && epoch==generation) occupied[slot]=false;
                    disposed=true;
                }
            }
        }
        @Override public Snapshot snapshot() throws IOException {
            synchronized(FlashNextDeltaNetExecutor.this) {
                requireOpen();int slot=0;while(slot<capacity && occupied[slot]) slot++;
                if(slot==capacity) throw new IllegalStateException("GPU DeltaNet checkpoint pool exhausted: "+capacity);
                execute(1,slot);occupied[slot]=true;checkpoints++;return new DeviceSnapshot(slot);
            }
        }
        @Override public void restore(Snapshot snapshot) throws IOException {
            synchronized(FlashNextDeltaNetExecutor.this) {
                requireOpen();
                if(!(snapshot instanceof DeviceSnapshot saved) || saved.owner!=this || saved.disposed || saved.epoch!=generation)
                    throw new IllegalArgumentException("Invalid GPU DeltaNet checkpoint owner/lifetime");
                execute(2,saved.slot);restores++;
            }
        }
        @Override public void reset() throws IOException {
            synchronized(FlashNextDeltaNetExecutor.this) {execute(3,0);generation++;Arrays.fill(occupied,false);resets++;}
        }
        @Override public void close() throws IOException {
            synchronized(FlashNextDeltaNetExecutor.this) {
                if(released) return;released=true;generation++;Arrays.fill(occupied,false);
                try {plan.close();} catch(Exception failure) {throw new IOException("Cannot release GPU DeltaNet",failure);}
                finally {arena.close();sessions.remove(this);resident-=bytes;}
            }
        }
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;
        IOException failure=null;
        for(var session:List.copyOf(sessions)) try {session.close();} catch(IOException closing) {
            if(failure==null) failure=closing;else failure.addSuppressed(closing);
        }
        closed=true;if(failure!=null) throw failure;
    }
}

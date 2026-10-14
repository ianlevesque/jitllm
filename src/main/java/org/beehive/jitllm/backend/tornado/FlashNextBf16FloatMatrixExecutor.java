package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** HC BF16 tensor-core projection with an FP32 accumulator/store and bounded residency. */
public final class FlashNextBf16FloatMatrixExecutor implements AutoCloseable {
    private static final int PAD=16;
    public record Statistics(long residentBytes,long peakBytes,long uploads,long evictions,long executions,long kernelNanos,long cacheHits,long cacheMisses,long hostUploadBytes,long hostDownloadBytes) {}
    private record Key(ModelOptMatrix matrix,int partition) {}
    private final TornadoDevice device;
    private final long budget;
    private final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,peak,uploads,evictions,executions,kernelNanos,graphId,cacheHits,cacheMisses,hostUploadBytes,hostDownloadBytes;
    private boolean closed;
    public FlashNextBf16FloatMatrixExecutor(int index,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive HC projection budget required");
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(index);this.budget=budget;
    }
    public synchronized Statistics statistics() {return new Statistics(resident,peak,uploads,evictions,executions,kernelNanos,cacheHits,cacheMisses,hostUploadBytes,hostDownloadBytes);}
    private static long size(Key key) {
        var m=key.matrix;return m.weights().bytes()+4L*(m.columns()+m.rows()+2L)
                +2L*PAD*m.columns()+4L*PAD*m.rows()*(m.columns()/key.partition)+6L*TornadoNativeArray.ARRAY_HEADER+FlashNextBf16FloatLibrary.WORKSPACE_BYTES;
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();final FloatArray input,output;final TornadoExecutionPlan plan;final long bytes;
        Entry(Key key) throws IOException {
            var m=key.matrix;bytes=size(key);TornadoExecutionPlan built=null;
            try {
                input=floats(m.columns());output=floats(m.rows());var weights=bfloats(Math.multiplyExact(m.rows(),m.columns()));
                try(var temporary=Arena.ofConfined();var file=FileChannel.open(m.weights().file(),StandardOpenOption.READ)) {
                    weights.getSegment().copyFrom(file.map(FileChannel.MapMode.READ_ONLY,m.weights().offset(),m.weights().bytes(),temporary));
                }
                var packed=bfloats(PAD*m.columns());var partials=floats(PAD*m.rows()*(m.columns()/key.partition));var control=floats(2);control.set(0,1);
                var graph=new TaskGraph("flashnextBfFloat"+(graphId++))
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,control,packed,partials)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,input)
                        .task("pack",FlashNextWeightKernels::bf16GemmInput,input,packed,m.columns(),1)
                        .libraryTask("gemm",FlashNextBf16FloatLibrary::partitions,m.rows(),PAD,m.columns(),weights,packed,control,partials,key.partition)
                        .task("reduce",FlashNextBf16FloatMatrixExecutor::reduce,partials,output,PAD,m.columns()/key.partition)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT);plan=built;
            } catch(Exception|Error failure) {if(built!=null) try {built.close();} catch(Exception closing) {failure.addSuppressed(closing);}arena.close();throw new IOException("Cannot build BF16/F32 HC projection",failure);}
        }
        FloatArray floats(int n) {return FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*n,32));}
        BFloat16Array bfloats(int n) {return BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+2L*n,32));}
        @Override public void close() throws IOException {try {plan.close();} catch(Exception failure) {throw new IOException("Cannot close BF16/F32 HC projection",failure);} finally {arena.close();}}
    }
    public static void reduce(FloatArray partials,FloatArray output,int batch,int groups) {
        for(@uk.ac.manchester.tornado.api.annotations.Parallel int row=0;row<output.getSize();row++) {
            float sum=partials.get(row);for(int group=1;group<groups;group++) sum+=partials.get(group*batch*output.getSize()+row);output.set(row,sum);
        }
    }
    public synchronized void multiply(ModelOptMatrix matrix,float[] values,float[] output,int partition) throws IOException {
        if(closed) throw new IOException("HC projection executor is closed");
        if(!matrix.dtype().equals("BF16") || matrix.rows()%16!=0 || partition<=0 || partition%16!=0 || matrix.columns()%partition!=0
                || values.length!=matrix.columns() || output.length!=matrix.rows()) throw new IllegalArgumentException("Invalid BF16/F32 projection");
        for(float value:values) if(!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite HC projection input");
        var key=new Key(matrix,partition);var entry=cache.get(key);
        if(entry!=null) cacheHits++;else cacheMisses++;
        if(entry==null) {
            long bytes=size(key);if(bytes>budget) throw new IOException("HC projection exceeds budget "+budget);
            while(bytes>budget-resident) {var iterator=cache.entrySet().iterator();var old=iterator.next();old.getValue().close();iterator.remove();resident-=old.getValue().bytes;evictions++;}
            entry=new Entry(key);cache.put(key,entry);resident+=entry.bytes;peak=Math.max(peak,resident);uploads++;
        }
        for(int i=0;i<values.length;i++) entry.input.set(i,values[i]);
        try {
            long time=entry.plan.execute().getProfilerResult().getDeviceKernelTime();if(time<=0) throw new IOException("HC BF16/F32 projection recorded no CUDA kernels");
            kernelNanos+=time;executions++;hostUploadBytes+=4L*values.length;hostDownloadBytes+=4L*output.length;for(int i=0;i<output.length;i++) output[i]=entry.output.get(i);
        } catch(RuntimeException failure) {throw new IOException("HC BF16/F32 projection failed",failure);}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        cache.clear();resident=0;if(failure!=null) throw failure;
    }
}

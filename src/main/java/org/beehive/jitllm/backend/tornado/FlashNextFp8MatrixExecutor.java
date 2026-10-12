package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextFp8Kernels;
import org.beehive.jitllm.format.safetensors.*;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Native W8A8 block-FP8 projections. Preserve packed weights and checkpoint scales;
 * distinct input rows are encoded on CUDA. Native K=128 partials stay on the device.
 */
public final class FlashNextFp8MatrixExecutor implements AutoCloseable {
    public record Statistics(long residentBytes,long uploads,long evictions,long executions,long kernelNanos,int matrices) {}
    private record Key(ModelOptMatrix matrix,int batch) {}
    private final TornadoDevice device;
    private final long budget;
    private final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,uploads,evictions,executions,kernelNanos,graphId;
    private boolean closed;
    public FlashNextFp8MatrixExecutor(int device,long budget) {this(TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(device),budget);}
    public FlashNextFp8MatrixExecutor(TornadoDevice device,long budget) {if(budget<=0) throw new IllegalArgumentException("Positive FP8 cache budget required");this.device=device;this.budget=budget;}
    public synchronized Statistics statistics() {return new Statistics(resident,uploads,evictions,executions,kernelNanos,cache.size());}
    private static long footprint(Key key) {
        var matrix=key.matrix;int batch=key.batch,k=matrix.columns(),n=matrix.rows();
        return matrix.weights().bytes()+4L*matrix.blockScales().elements()+5L*batch*k
                +4L*batch*k/128+4L*batch*n*(k/128+1)+127*4L+16+9L*TornadoNativeArray.ARRAY_HEADER+FlashNextFp8Library.WORKSPACE_BYTES;
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();final long bytes;final int rows,columns,batch;
        final FloatArray input,inputScales,control,output;final ByteArray encoded;final TornadoExecutionPlan plan;
        Entry(Key key) throws Exception {
            rows=key.matrix.rows();columns=key.matrix.columns();batch=key.batch;bytes=footprint(key);TornadoExecutionPlan built=null;
            try {
                var weights=bytes(Math.toIntExact(key.matrix.weights().bytes()));
                try(var mapping=Arena.ofConfined();var file=FileChannel.open(key.matrix.weights().file(),StandardOpenOption.READ)) {
                    weights.getSegment().copyFrom(file.map(FileChannel.MapMode.READ_ONLY,key.matrix.weights().offset(),key.matrix.weights().bytes(),mapping));
                }
                var scales=floats(Math.toIntExact(key.matrix.blockScales().elements()));
                var scaleTensor=key.matrix.blockScales();byte[] raw=new byte[Math.toIntExact(scaleTensor.bytes())];
                try(var file=FileChannel.open(scaleTensor.file(),StandardOpenOption.READ)) {SafeTensorIndex.readFully(file,ByteBuffer.wrap(raw),scaleTensor.offset());}
                var buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for(int i=0;i<scales.getSize();i++) scales.set(i,scaleTensor.dtype().equals("F32") ? buffer.getFloat() : FloatFormats.bf16(buffer.getShort()));
                input=floats(batch*columns);inputScales=floats(batch*columns/128);encoded=bytes(batch*columns);
                var partials=floats(batch*rows*(columns/128));output=floats(batch*rows);control=floats(4);control.set(0,1);control.set(2,1);control.set(3,1);
                var table=floats(127);for(int i=0;i<127;i++) table.set(i,FloatFormats.e4m3(i));
                var graph=new TaskGraph("flashnextFp8"+(graphId++)).transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,scales,table)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,input,control)
                        .task("encode",FlashNextFp8Kernels::quantize,input,table,encoded,inputScales)
                        .libraryTask("partitions",FlashNextFp8Library::partitions,rows,batch,columns,weights,encoded,control,partials)
                        .task("reduce",FlashNextFp8Kernels::reduce,partials,scales,inputScales,control,output,rows,columns)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output).transferToHost(DataTransferMode.UNDER_DEMAND,encoded,inputScales);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT).withCompilerFlags(TornadoVMBackendType.CUDA,"--use_fast_math");plan=built;
            } catch(Exception|Error failure) {if(built!=null) try {built.close();} catch(Exception closing) {failure.addSuppressed(closing);}arena.close();throw failure;}
        }
        ByteArray bytes(int n) {return ByteArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+(long)n,32));}
        FloatArray floats(int n) {return FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*n,32));}
        public void close() throws Exception {try {plan.close();} finally {arena.close();}}
    }
    private Entry entry(Key key) throws IOException {
        if(closed) throw new IOException("FP8 executor is closed");var entry=cache.get(key);if(entry!=null) return entry;
        long size=footprint(key);if(size>budget) throw new IOException("FP8 matrix/workspace exceeds cache budget "+budget);
        while(size>budget-resident) {
            var iterator=cache.entrySet().iterator();var oldest=iterator.next();
            try {oldest.getValue().close();} catch(Exception failure) {throw new IOException("Cannot evict FP8 matrix",failure);}
            iterator.remove();resident-=oldest.getValue().bytes;evictions++;
        }
        try {entry=new Entry(key);} catch(Exception failure) {throw new IOException("Cannot build FP8 GEMM",failure);}
        cache.put(key,entry);resident+=entry.bytes;uploads++;return entry;
    }
    public synchronized void multiply(ModelOptMatrix matrix,float[] values,float[] result,int rows,float routeWeight) throws IOException {
        if(!matrix.dtype().equals("F8_E4M3") || matrix.columns()%128!=0 || matrix.rows()%128!=0 || rows<=0
                || values.length!=Math.multiplyExact(rows,matrix.columns()) || result.length!=Math.multiplyExact(rows,matrix.rows()) || !Float.isFinite(routeWeight))
            throw new IllegalArgumentException("Invalid block FP8 matrix/input");
        var entry=entry(new Key(matrix,Math.multiplyExact((rows+31)/32,32)));entry.input.getSegment().fill((byte)0);
        for(int i=0;i<values.length;i++) {if(!Float.isFinite(values[i])) throw new IllegalArgumentException("Nonfinite FP8 activation");entry.input.set(i,FloatFormats.roundBf16(values[i]));}
        entry.control.set(3,routeWeight);
        try {var execution=entry.plan.execute();long time=execution.getProfilerResult().getDeviceKernelTime();if(time<=0) throw new IOException("FP8 recorded no CUDA kernels");
            for(int i=0;i<result.length;i++) result[i]=entry.output.get(i);executions++;kernelNanos+=time;
        } catch(RuntimeException failure) {throw new IOException("FP8 execution failed",failure);}
    }
    public synchronized byte[] readEncoded(ModelOptMatrix matrix,int rows) throws IOException {
        var entry=entry(new Key(matrix,Math.multiplyExact((rows+31)/32,32)));entry.plan.execute().transferToHost(entry.encoded);byte[] data=new byte[rows*matrix.columns()];
        for(int i=0;i<data.length;i++) data[i]=entry.encoded.get(i);return data;
    }
    public synchronized float[] readScales(ModelOptMatrix matrix,int rows) throws IOException {
        var entry=entry(new Key(matrix,Math.multiplyExact((rows+31)/32,32)));entry.plan.execute().transferToHost(entry.inputScales);float[] data=new float[rows*matrix.columns()/128];
        for(int i=0;i<data.length;i++) data[i]=entry.inputScales.get(i);return data;
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(Exception closing) {if(failure==null) failure=new IOException("Cannot close FP8 cache",closing);else failure.addSuppressed(closing);}
        cache.clear();resident=0;if(failure!=null) throw failure;
    }
}

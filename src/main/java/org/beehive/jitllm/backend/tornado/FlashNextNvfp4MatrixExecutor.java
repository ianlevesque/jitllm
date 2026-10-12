package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels;
import org.beehive.jitllm.format.safetensors.*;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Bounded native packed NVFP4 GEMM. Batch inputs are distinct quantized rows;
 * padded rows are zero. No weight or activation is expanded to BF16 before GEMM.
 */
public final class FlashNextNvfp4MatrixExecutor implements AutoCloseable {
    public record Statistics(long residentBytes,long uploads,long evictions,long executions,long kernelNanos,int matrices) {}
    private record Key(ModelOptMatrix matrix,int paddedBatch) {}
    private final TornadoDevice device;
    private final long budget;
    private final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,uploads,evictions,executions,kernelNanos,graphId;
    private boolean closed;
    public FlashNextNvfp4MatrixExecutor(int device,long budget) {this(TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(device),budget);}
    public FlashNextNvfp4MatrixExecutor(TornadoDevice device,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive native NVFP4 budget required");this.device=device;this.budget=budget;
    }
    public synchronized Statistics statistics() {return new Statistics(resident,uploads,evictions,executions,kernelNanos,cache.size());}
    private static int padded(int rows) {if(rows<=0) throw new IllegalArgumentException("Positive NVFP4 batch required");return Math.multiplyExact((Math.addExact(rows,31)/32),32);}
    private static long scales(int rows,int columns) {return Math.multiplyExact(((long)rows+127)/128*128,columns/16);}
    private static long bytes(Key key) {
        var matrix=key.matrix;long batch=key.paddedBatch,columns=matrix.columns(),rows=matrix.rows();
        return matrix.weights().bytes()+matrix.blockScales().bytes()+scales(matrix.rows(),matrix.columns())
                +batch*columns/2+batch*columns/16+scales(key.paddedBatch,matrix.columns())+6*batch*rows
                +4L*batch*columns+4L*(256+8)+13L*TornadoNativeArray.ARRAY_HEADER+32+FlashNextNvfp4Library.WORKSPACE_BYTES;
    }
    private static ByteArray mapped(SafeTensorIndex.Tensor tensor,Arena arena) throws IOException {
        long header=TornadoNativeArray.ARRAY_HEADER;
        boolean shallow=tensor.offset()>=header && tensor.offset()%8==0 && Files.isWritable(tensor.file());
        var options=shallow ? new StandardOpenOption[]{StandardOpenOption.READ,StandardOpenOption.WRITE} : new StandardOpenOption[]{StandardOpenOption.READ};
        try(var channel=FileChannel.open(tensor.file(),options)) {
            if(shallow) return ByteArray.fromSegmentShallow(channel.map(FileChannel.MapMode.PRIVATE,tensor.offset()-header,tensor.bytes()+header,arena));
            var segment=arena.allocate(tensor.bytes()+header,32);
            segment.asSlice(header,tensor.bytes()).copyFrom(channel.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),arena));
            return ByteArray.fromSegmentShallow(segment);
        }
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();
        final ByteArray packed,rawScales;
        final FloatArray alphaBeta,output,rawInput;
        final IntArray initialized;
        final TornadoExecutionPlan plan;
        TornadoExecutionResult lastResult;
        final long bytes;
        Entry(Key key) throws Exception {
            bytes=bytes(key);TornadoExecutionPlan built=null;
            try {
                var matrix=key.matrix;int batch=key.paddedBatch,columns=matrix.columns(),rows=matrix.rows();
                var weights=mapped(matrix.weights(),arena);var weightScales=mapped(matrix.blockScales(),arena);
                var swizzledWeights=byteArray(Math.toIntExact(scales(rows,columns)));
                packed=byteArray(Math.multiplyExact(batch,columns)/2);rawScales=byteArray(Math.multiplyExact(batch,columns)/16);
                var swizzledInput=byteArray(Math.toIntExact(scales(batch,columns)));
                alphaBeta=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+12,32));
                initialized=IntArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+20,32));
                initialized.set(2,127);initialized.set(3,8);initialized.set(4,2);
                rawInput=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*batch*columns,32));
                var e4=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*256,32));
                var e2=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*8,32));
                for(int i=0;i<256;i++) e4.set(i,FloatFormats.e4m3(i));for(int i=0;i<8;i++) e2.set(i,FloatFormats.e2m1(i));
                var result=BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+2L*batch*rows,32));
                output=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*batch*rows,32));
                var graph=new TaskGraph("flashnextNvfp4"+(graphId++))
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,weightScales,swizzledWeights,swizzledInput,result,e4,e2)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,packed,rawScales,alphaBeta,initialized,rawInput)
                        .task("quantize",FlashNextWeightKernels::quantizeNvfp4,rawInput,alphaBeta,initialized,e4,e2,packed,rawScales)
                        .task("weightScales",FlashNextWeightKernels::swizzleNvfp4Scales,weightScales,swizzledWeights,initialized,rows,columns/16,1)
                        .task("inputScales",FlashNextWeightKernels::swizzleNvfp4Scales,rawScales,swizzledInput,initialized,batch,columns/16,0)
                        .libraryTask("gemm",FlashNextNvfp4Library::gemm,rows,batch,columns,weights,swizzledWeights,packed,swizzledInput,alphaBeta,result)
                        .task("unpack",FlashNextWeightKernels::bf16GemmOutput,result,output)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output).transferToHost(DataTransferMode.UNDER_DEMAND,packed,rawScales);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT)
                        .withCompilerFlags(TornadoVMBackendType.CUDA,"--prec-div=false");plan=built;
            } catch(Exception|Error failure) {if(built!=null) try {built.close();} catch(Exception cleanup) {failure.addSuppressed(cleanup);}arena.close();throw failure;}
        }
        ByteArray byteArray(int count) {return ByteArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+(long)count,32));}
        @Override public void close() throws Exception {try {plan.close();} finally {arena.close();}}
    }
    private Entry entry(Key key) throws IOException {
        if(closed) throw new IOException("Native NVFP4 cache is closed");var entry=cache.get(key);if(entry!=null) return entry;
        long bytes=bytes(key);if(bytes>budget) throw new IOException("Native NVFP4 matrix/workspace exceeds cache budget "+budget);
        while(bytes>budget-resident) {
            var iterator=cache.entrySet().iterator();var oldest=iterator.next();
            try {oldest.getValue().close();} catch(Exception failure) {throw new IOException("Cannot evict native NVFP4 plan",failure);}
            resident-=oldest.getValue().bytes;iterator.remove();evictions++;
        }
        try {entry=new Entry(key);} catch(Exception failure) {throw new IOException("Cannot build native NVFP4 GEMM",failure);}
        cache.put(key,entry);resident+=entry.bytes;uploads++;return entry;
    }
    public synchronized void multiplyRow(ModelOptMatrix matrix,Nvfp4Activation.Packed input,float[] output) throws IOException {multiplyBatch(matrix,new Nvfp4Activation.Packed[]{input},output);}
    public synchronized void multiplyBatch(ModelOptMatrix matrix,Nvfp4Activation.Packed[] inputs,float[] output) throws IOException {
        if(!matrix.dtype().equals("U8") || output.length!=Math.multiplyExact(inputs.length,matrix.rows())) throw new IllegalArgumentException("Invalid native NVFP4 matrix/output");
        int batch=padded(inputs.length),columns=matrix.columns();float scale=inputs[0].globalScale();
        for(var input:inputs) if(input.values().length!=columns/2 || input.scales().length!=columns/16 || input.globalScale()!=scale || !Float.isFinite(scale) || scale<=0)
            throw new IllegalArgumentException("Invalid native NVFP4 input dimensions/scales");
        var entry=entry(new Key(matrix,batch));entry.packed.getSegment().fill((byte)0);entry.rawScales.getSegment().fill((byte)0);
        for(int row=0;row<inputs.length;row++) {
            entry.packed.getSegment().asSlice((long)row*columns/2,columns/2).copyFrom(java.lang.foreign.MemorySegment.ofArray(inputs[row].values()));
            entry.rawScales.getSegment().asSlice((long)row*columns/16,columns/16).copyFrom(java.lang.foreign.MemorySegment.ofArray(inputs[row].scales()));
        }
        entry.alphaBeta.set(0,matrix.globalScale()*scale);entry.alphaBeta.set(1,0);
        entry.initialized.set(1,0);
        run(entry,output);
    }
    public synchronized void multiplyValues(ModelOptMatrix matrix,float[] input,float scale,float[] output,int batch) throws IOException {
        if(!matrix.dtype().equals("U8") || input.length!=Math.multiplyExact(batch,matrix.columns()) || output.length!=Math.multiplyExact(batch,matrix.rows())
                || !Float.isFinite(scale) || scale<=0 || !Float.isFinite(1f/scale)) throw new IllegalArgumentException("Invalid native NVFP4 input");
        for(float value:input) if(!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite native NVFP4 input");
        var entry=entry(new Key(matrix,padded(batch)));entry.rawInput.getSegment().fill((byte)0);
        for(int i=0;i<input.length;i++) entry.rawInput.set(i,input[i]);
        entry.alphaBeta.set(0,matrix.globalScale()*scale);entry.alphaBeta.set(1,0);entry.alphaBeta.set(2,1f/scale);entry.initialized.set(1,1);
        run(entry,output);
    }
    /** Explicit diagnostics only; normal projections never download their quantized activations. */
    public synchronized byte[][] readPacked(ModelOptMatrix matrix,int batch) {
        var entry=cache.get(new Key(matrix,padded(batch)));if(entry==null) throw new IllegalStateException("Native NVFP4 batch is not cached");
        entry.lastResult.transferToHost(entry.packed,entry.rawScales);
        byte[] values=new byte[batch*matrix.columns()/2],scales=new byte[batch*matrix.columns()/16];
        java.lang.foreign.MemorySegment.ofArray(values).copyFrom(entry.packed.getSegment().asSlice(0,values.length));
        java.lang.foreign.MemorySegment.ofArray(scales).copyFrom(entry.rawScales.getSegment().asSlice(0,scales.length));
        return new byte[][]{values,scales};
    }
    private void run(Entry entry,float[] output) throws IOException {
        try {
            var result=entry.plan.execute();long nanos=result.getProfilerResult().getDeviceKernelTime();if(nanos<=0) throw new IOException("Native NVFP4 recorded no CUDA kernel");
            entry.lastResult=result;
            for(int i=0;i<output.length;i++) output[i]=entry.output.get(i);
            entry.initialized.set(0,1);kernelNanos+=nanos;executions++;
        } catch(RuntimeException failure) {throw new IOException("Native NVFP4 GEMM failed",failure);}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(Exception closing) {if(failure==null) failure=new IOException("Cannot close native NVFP4 cache",closing);else failure.addSuppressed(closing);}
        cache.clear();resident=0;if(failure!=null) throw failure;
    }
}

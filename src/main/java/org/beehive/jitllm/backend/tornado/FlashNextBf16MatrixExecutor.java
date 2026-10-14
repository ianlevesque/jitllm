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
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.CuBlasOptions;

/** Native BF16 GEMM with the real batch geometry. The sequential reference runner broadcasts
 * one row; multiplyBatch is the same primitive for a future batched prefill plan without that work.
 * Packed expert weights remain in the separate NVFP4 cache. This cache holds only BF16 matrices.
 */
public final class FlashNextBf16MatrixExecutor implements AutoCloseable {
    public record Statistics(long residentBytes,long uploads,long evictions,long executions,long kernelNanos,long cacheHits,long cacheMisses,long hostUploadBytes,long hostDownloadBytes) {}
    private static final long WORKSPACE=32L<<20;
    // The pinned SDK rebinds a cuBLAS handle when its caller thread changes, which
    // resets the native workspace. Give each caller its own plan and handle.
    private record Key(ModelOptMatrix matrix,int batch,boolean broadcast,long thread) {}
    private final TornadoDevice device;
    private final long budget;
    private final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,uploads,evictions,executions,kernelNanos,graphId,cacheHits,cacheMisses,hostUploadBytes,hostDownloadBytes;
    private boolean closed;
    public FlashNextBf16MatrixExecutor(TornadoDevice device,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive BF16 GEMM budget required");
        this.device=device;this.budget=budget;
    }
    public FlashNextBf16MatrixExecutor(int deviceIndex,long budget) {this(TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex),budget);}
    public synchronized Statistics statistics() {return new Statistics(resident,uploads,evictions,executions,kernelNanos,cacheHits,cacheMisses,hostUploadBytes,hostDownloadBytes);}
    private static long size(Key key) {
        long columns=key.matrix().columns(),rows=key.matrix().rows();
        long source=key.broadcast() ? columns : Math.multiplyExact(columns,key.batch());
        long result=key.broadcast() ? rows : Math.multiplyExact(rows,key.batch());
        return key.matrix().weights().bytes()+4*(source+result)+2L*key.batch()*(rows+columns)+5L*TornadoNativeArray.ARRAY_HEADER+WORKSPACE;
    }
    private static LibraryTaskDescriptor gemm(int ta,int tb,int m,int n,int k,float alpha,BFloat16Array a,int lda,BFloat16Array b,int ldb,float beta,BFloat16Array c,int ldc) {
        return CuBlas.cublasGemmExBF16(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc).withTuning(new CuBlasOptions().withWorkspace(WORKSPACE));
    }
    private static BFloat16Array mapped(SafeTensorIndex.Tensor tensor,Arena arena) throws IOException {
        long header=TornadoNativeArray.ARRAY_HEADER;
        boolean shallow=tensor.offset()>=header && tensor.offset()%Long.BYTES==0 && Files.isWritable(tensor.file());
        var options=shallow ? new StandardOpenOption[]{StandardOpenOption.READ,StandardOpenOption.WRITE} : new StandardOpenOption[]{StandardOpenOption.READ};
        try(var channel=FileChannel.open(tensor.file(),options)) {
            if(shallow) return BFloat16Array.fromSegmentShallow(channel.map(FileChannel.MapMode.PRIVATE,tensor.offset()-header,tensor.bytes()+header,arena));
            var raw=channel.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),arena);
            var segment=arena.allocate(tensor.bytes()+header,32);segment.asSlice(header,tensor.bytes()).copyFrom(raw);
            return BFloat16Array.fromSegmentShallow(segment);
        }
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();
        final FloatArray input,output;
        final TornadoExecutionPlan plan;
        final long bytes;
        Entry(Key key) throws Exception {
            bytes=size(key);var matrix=key.matrix();
            try {
                input=floats(key.broadcast() ? matrix.columns() : Math.multiplyExact(matrix.columns(),key.batch()));
                output=floats(key.broadcast() ? matrix.rows() : Math.multiplyExact(matrix.rows(),key.batch()));
                var weights=mapped(matrix.weights(),arena);
                var expanded=bfloats(Math.multiplyExact(matrix.columns(),key.batch()));
                var result=bfloats(Math.multiplyExact(matrix.rows(),key.batch()));
                var graph=new TaskGraph("flashnextBfGemm"+(graphId++))
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,expanded,result)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,input)
                        .task("pack",FlashNextWeightKernels::bf16GemmInput,input,expanded,matrix.columns(),key.broadcast() ? 1 : 0)
                        // Row-major W[R,K] is a column-major [K,R] view; X[M,K] is [K,M].
                        .libraryTask("gemm",FlashNextBf16MatrixExecutor::gemm,1,0,matrix.rows(),key.batch(),matrix.columns(),1f,
                                weights,matrix.columns(),expanded,matrix.columns(),0f,result,matrix.rows())
                        .task("unpack",FlashNextWeightKernels::bf16GemmOutput,result,output)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION,output);
                plan=new TornadoExecutionPlan(graph.snapshot()).withDevice(device).withProfiler(ProfilerMode.SILENT);
            } catch(Exception|Error failure) {arena.close();throw failure;}
        }
        FloatArray floats(int count) {return FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*count,32));}
        BFloat16Array bfloats(int count) {return BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+2L*count,32));}
        @Override public void close() throws Exception {try {plan.close();} finally {arena.close();}}
    }
    public synchronized void multiplyRow(ModelOptMatrix matrix,float[] input,float[] output,int prefillRows) throws IOException {
        multiply(new Key(matrix,prefillRows,true,Thread.currentThread().threadId()),input,output);
    }
    public synchronized void multiplyBatch(ModelOptMatrix matrix,float[] input,float[] output,int batchRows) throws IOException {
        multiply(new Key(matrix,batchRows,false,Thread.currentThread().threadId()),input,output);
    }
    private void multiply(Key key,float[] input,float[] output) throws IOException {
        if(closed) throw new IOException("BF16 GEMM cache is closed");
        if(key.batch()<=0 || !key.matrix().dtype().equals("BF16")) throw new IllegalArgumentException("BF16 GEMM requires BF16 weights and positive batch rows");
        int in=key.broadcast() ? key.matrix().columns() : Math.multiplyExact(key.matrix().columns(),key.batch());
        int out=key.broadcast() ? key.matrix().rows() : Math.multiplyExact(key.matrix().rows(),key.batch());
        if(input.length!=in || output.length!=out) throw new IllegalArgumentException("Wrong BF16 GEMM input/output shape");
        var entry=cache.get(key);
        if(entry!=null) cacheHits++;else cacheMisses++;
        if(entry==null) {
            long bytes=size(key);if(bytes>budget) throw new IOException("BF16 batch GEMM requires "+bytes+" bytes, exceeding budget "+budget);
            while(resident+bytes>budget) {
                var iterator=cache.entrySet().iterator();var oldest=iterator.next();
                try {oldest.getValue().close();} catch(Exception failure) {throw new IOException("Cannot evict BF16 GEMM",failure);}
                resident-=oldest.getValue().bytes;iterator.remove();evictions++;
            }
            try {entry=new Entry(key);} catch(Exception failure) {throw new IOException("Cannot build BF16 GEMM",failure);}
            cache.put(key,entry);resident+=entry.bytes;uploads++;
        }
        for(int i=0;i<input.length;i++) entry.input.set(i,input[i]);
        try {
            long nanos=entry.plan.execute().getProfilerResult().getDeviceKernelTime();
            if(nanos<=0) throw new IOException("BF16 GEMM packing/unpacking kernels did not execute");
            executions++;kernelNanos+=nanos;hostUploadBytes+=4L*input.length;hostDownloadBytes+=4L*output.length;for(int i=0;i<output.length;i++) output[i]=entry.output.get(i);
        } catch(RuntimeException failure) {throw new IOException("Native BF16 GEMM failed",failure);}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(Exception closing) {
            if(failure==null) failure=new IOException("Cannot close BF16 GEMM cache",closing);else failure.addSuppressed(closing);
        }
        cache.clear();resident=0;if(failure!=null) throw failure;
    }
}

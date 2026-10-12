package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels;
import org.beehive.jitllm.format.safetensors.*;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/** Bounded, storage-preserving GPU projections for the full host layer pipeline. */
public final class FlashNextMatrixExecutor implements MatrixExecutor,AutoCloseable {
    public record Statistics(long residentBytes,long uploads,long evictions,long executions,long kernelNanos,int matrices,long bf16ResidentBytes,long bf16Executions) {}
    private final TornadoDevice device;
    private final long budget;
    private final FlashNextBf16MatrixExecutor bf16;
    private final LinkedHashMap<ModelOptMatrix,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    private long resident,uploads,evictions,executions,kernelNanos,graphId;
    private boolean closed;

    public FlashNextMatrixExecutor(int deviceIndex,long budget) {
        if(budget<=0) throw new IllegalArgumentException("Positive GPU weight cache budget required");
        this.device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);
        this.budget=budget;
        bf16=new FlashNextBf16MatrixExecutor(device,Math.min(budget,256L<<20));
    }
    public String deviceName() { return device.getPhysicalDevice().getDeviceName(); }
    public synchronized Statistics statistics() { var extra=bf16.statistics();return new Statistics(resident,uploads,evictions,executions,kernelNanos,cache.size(),extra.residentBytes(),extra.executions()); }

    // PRIVATE mappings let Tornado's array header occupy a private copy of the preceding
    // bytes. No checkpoint bytes are changed, and no full host weight copy is required.
    private static ByteArray mapped(SafeTensorIndex.Tensor tensor,Arena arena) throws IOException {
        long header=TornadoNativeArray.ARRAY_HEADER;
        boolean shallow=tensor.offset()>=header && tensor.offset()%Long.BYTES==0 && java.nio.file.Files.isWritable(tensor.file());
        var options=shallow ? new StandardOpenOption[]{StandardOpenOption.READ,StandardOpenOption.WRITE} : new StandardOpenOption[]{StandardOpenOption.READ};
        try(var channel=FileChannel.open(tensor.file(),options)) {
            if(shallow) {
                var segment=channel.map(FileChannel.MapMode.PRIVATE,tensor.offset()-header,Math.addExact(tensor.bytes(),header),arena);
                return ByteArray.fromSegmentShallow(segment);
            }
            var data=channel.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),arena);
            var segment=arena.allocate(Math.addExact(tensor.bytes(),header),Long.BYTES);
            segment.asSlice(header,tensor.bytes()).copyFrom(data);
            return ByteArray.fromSegmentShallow(segment);
        }
    }
    private final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();
        final long bytes;
        final FloatArray input,output,alpha;
        final ByteArray packedInput,inputScales;
        final IntArray mode;
        final TornadoExecutionPlan plain;
        Entry(ModelOptMatrix matrix) throws Exception {
            bytes=matrix.weights().bytes()+(matrix.blockScales()==null ? 0 : matrix.blockScales().bytes())
                    +4L*(matrix.columns()+matrix.rows()+273)+matrix.columns()/2+matrix.columns()/16;
            TornadoExecutionPlan plainPlan=null;
            try {
                var weights=mapped(matrix.weights(),arena);
                input=new FloatArray(matrix.columns());output=new FloatArray(matrix.rows());alpha=new FloatArray(1);
                packedInput=new ByteArray(matrix.columns()/2);inputScales=new ByteArray(matrix.columns()/16);
                mode=new IntArray(1);
                var e2m1=new FloatArray(16);var e4m3=new FloatArray(256);
                for(int i=0;i<16;i++) e2m1.set(i,FloatFormats.e2m1(i));
                for(int i=0;i<256;i++) e4m3.set(i,FloatFormats.e4m3(i));
                String id="flashnextMatrix"+(graphId++);
                var graph=new TaskGraph(id).transferToDevice(DataTransferMode.EVERY_EXECUTION,input);
                if(matrix.dtype().equals("U8")) {
                    var scales=mapped(matrix.blockScales(),arena);
                    alpha.set(0,matrix.globalScale());
                    graph.transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,scales,e2m1,e4m3)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION,packedInput,inputScales,alpha,mode)
                            .task("project",FlashNextWeightKernels::nvfp4Projection,input,packedInput,inputScales,mode,weights,scales,e2m1,e4m3,alpha,output,matrix.columns());
                } else if(matrix.dtype().equals("F8_E4M3")) {
                    var tensor=matrix.blockScales();
                    MemorySegment raw;
                    try(var channel=FileChannel.open(tensor.file(),StandardOpenOption.READ)) {
                        raw=channel.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),arena);
                    }
                    var scales=new FloatArray(Math.toIntExact(tensor.elements()));
                    var bytes=raw.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                    for(int i=0;i<scales.getSize();i++) scales.set(i,tensor.dtype().equals("BF16") ? FloatFormats.bf16(bytes.getShort()) : bytes.getFloat());
                    graph.transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,scales,e4m3)
                            .task("project",FlashNextWeightKernels::blockFp8Matvec,input,weights,scales,e4m3,output,matrix.columns());
                } else {
                    int dtype=switch(matrix.dtype()) {case "BF16"->0;case "F16"->1;case "F32"->2;default->throw new IOException("Unsupported GPU matrix dtype "+matrix.dtype());};
                    graph.transferToDevice(DataTransferMode.FIRST_EXECUTION,weights)
                            .task("project",FlashNextWeightKernels::denseMatvec,input,weights,output,matrix.columns(),dtype);
                }
                plainPlan=new TornadoExecutionPlan(graph.transferToHost(DataTransferMode.EVERY_EXECUTION,output).snapshot())
                        .withDevice(device).withProfiler(ProfilerMode.SILENT);
                plain=plainPlan;
            } catch(Exception|Error failure) {
                if(plainPlan!=null) plainPlan.close();arena.close();throw failure;
            }
        }
        public void close() throws Exception {
            try {plain.close();} finally {arena.close();}
        }
    }
    private Entry entry(ModelOptMatrix matrix) throws IOException {
        if(closed) throw new IOException("GPU projection cache is closed");
        Entry result=cache.get(matrix);if(result!=null) return result;
        long size=matrix.weights().bytes()+(matrix.blockScales()==null ? 0 : matrix.blockScales().bytes())+4L*(matrix.columns()+matrix.rows()+273)+matrix.columns()/2+matrix.columns()/16;
        if(size>budget) throw new IOException("Matrix "+matrix.weights().name()+" needs "+size+" bytes, exceeding GPU cache budget "+budget);
        while(resident+size>budget) {
            var iterator=cache.entrySet().iterator();var oldest=iterator.next();
            try {oldest.getValue().close();} catch(Exception failure) {throw new IOException("Cannot evict GPU matrix",failure);}
            resident-=oldest.getValue().bytes;iterator.remove();evictions++;
        }
        try {result=new Entry(matrix);} catch(Exception failure) {throw new IOException("Cannot build GPU projection "+matrix.weights().name(),failure);}
        cache.put(matrix,result);resident+=result.bytes;uploads++;
        return result;
    }
    @Override public synchronized void multiply(ModelOptMatrix matrix,float[] values,float[] output) throws IOException {
        var entry=entry(matrix);
        entry.alpha.set(0,matrix.globalScale());
        entry.mode.set(0,0);
        for(int i=0;i<values.length;i++) entry.input.set(i,values[i]);
        run(entry.plain,entry,output);
    }
    @Override public synchronized void multiplyNvfp4(ModelOptMatrix matrix,Nvfp4Activation.Packed values,float[] output) throws IOException {
        var entry=entry(matrix);
        for(int i=0;i<values.values().length;i++) entry.packedInput.set(i,values.values()[i]);
        for(int i=0;i<values.scales().length;i++) entry.inputScales.set(i,values.scales()[i]);
        entry.alpha.set(0,matrix.globalScale()*values.globalScale());
        entry.mode.set(0,1);
        run(entry.plain,entry,output);
    }
    @Override public synchronized void multiplyBf16(ModelOptMatrix matrix,float[] input,float[] output,int prefillRows) throws IOException {
        if(closed) throw new IOException("GPU projection cache is closed");
        if(matrix.dtype().equals("BF16")) bf16.multiplyRow(matrix,input,output,prefillRows);
        else MatrixExecutor.super.multiplyBf16(matrix,input,output,prefillRows);
    }
    private void run(TornadoExecutionPlan plan,Entry entry,float[] output) throws IOException {
        try {
            long nanos=plan.execute().getProfilerResult().getDeviceKernelTime();
            if(nanos<=0) throw new IOException("GPU projection recorded no kernel execution");
            kernelNanos+=nanos;executions++;
            for(int i=0;i<output.length;i++) output[i]=entry.output.get(i);
        } catch(RuntimeException failure) {throw new IOException("GPU projection failed",failure);}
    }
    @Override public synchronized void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(Exception closing) {
            if(failure==null) failure=new IOException("Cannot close GPU projection cache",closing);else failure.addSuppressed(closing);
        }
        cache.clear();resident=0;
        try {bf16.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        if(failure!=null) throw failure;
    }
}

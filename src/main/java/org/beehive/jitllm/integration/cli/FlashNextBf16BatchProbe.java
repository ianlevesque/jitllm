package org.beehive.jitllm.integration.cli;

import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.tornado.FlashNextBf16MatrixExecutor;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.server.Json;

/** Distinct-row BF16 tensor-core GEMM, cache replay/thread ownership, and source-byte preservation. */
public final class FlashNextBf16BatchProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextBf16BatchProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));String prefix=(String)request.get("matrix");
        var index=SafeTensorIndex.open(Path.of(args[0]),name->name.equals(prefix+".weight"));
        int batch=((Number)request.get("batch_rows")).intValue(),device=((Number)request.getOrDefault("device",1)).intValue();
        try(var reader=new PagedTensorReader(1048576,4096);var executor=new FlashNextBf16MatrixExecutor(device,256L<<20)) {
            var matrix=new ModelOptMatrix(index,reader,prefix);var tensor=matrix.weights();
            byte[] before=window(tensor.file(),tensor.offset()-32),tail=window(tensor.file(),tensor.offset()+tensor.bytes()-64);
            var values=(List<?>)request.get("input");float[] input=new float[values.size()];for(int i=0;i<input.length;i++) input[i]=((Number)values.get(i)).floatValue();
            float[] output=new float[Math.multiplyExact(batch,matrix.rows())],replay=new float[output.length];
            executor.multiplyBatch(matrix,input,output,batch);executor.multiplyBatch(matrix,input,replay,batch);
            if(!Arrays.equals(output,replay)) throw new AssertionError("BF16 GEMM cache replay differs");
            var other=new FutureTask<float[]>(()-> {float[] result=new float[output.length];executor.multiplyBatch(matrix,input,result,batch);return result;});
            Thread.ofPlatform().start(other);
            if(!Arrays.equals(output,other.get())) throw new AssertionError("BF16 GEMM caller-thread result differs");
            executor.multiplyBatch(matrix,input,replay,batch);
            if(!Arrays.equals(output,replay)) throw new AssertionError("Returning BF16 GEMM caller lost its workspace");
            if(!Arrays.equals(before,window(tensor.file(),tensor.offset()-32)) || !Arrays.equals(tail,window(tensor.file(),tensor.offset()+tensor.bytes()-64)))
                throw new AssertionError("Native BF16 GEMM changed checkpoint bytes");
            var stats=executor.statistics();executor.close();
            if(executor.statistics().residentBytes()!=0) throw new AssertionError("BF16 GEMM cache retained buffers/workspace");
            var result=new LinkedHashMap<String,Object>();var encoded=new ArrayList<Double>();for(float value:output) encoded.add((double)value);
            result.put("output",encoded);result.put("batch_rows",batch);result.put("matrix",prefix);result.put("cache_replay_exact",true);
            result.put("caller_thread_exact",true);result.put("source_bytes_unchanged",true);result.put("resident_bytes_after_close",0);
            result.put("resident_bytes",stats.residentBytes());result.put("uploads",stats.uploads());result.put("executions",stats.executions());result.put("kernel_ns",stats.kernelNanos());
            System.out.println(Json.write(result));
        }
    }
    private static byte[] window(Path path,long offset) throws Exception {
        var bytes=ByteBuffer.allocate(64);try(var file=FileChannel.open(path,StandardOpenOption.READ)) {
            while(bytes.hasRemaining()) {int n=file.read(bytes,offset+bytes.position());if(n<=0) throw new IllegalStateException("Truncated checkpoint window");}
        }
        return bytes.array();
    }
}

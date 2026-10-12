package org.beehive.jitllm.integration.cli;

import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4MatrixExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4Library;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.server.Json;

/** Real packed NVFP4 tensor-core GEMM, distinct rows and native resource lifecycle. */
public final class FlashNextNvfp4BatchProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextNvfp4BatchProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));String prefix=(String)request.get("matrix");
        int batch=((Number)request.get("batch_rows")).intValue(),device=((Number)request.getOrDefault("device",1)).intValue();
        float scale=((Number)request.get("input_scale")).floatValue();
        var index=SafeTensorIndex.open(Path.of(args[0]),name->name.startsWith(prefix+"."));
        try(var reader=new PagedTensorReader(1048576,4096);var executor=new FlashNextNvfp4MatrixExecutor(device,256L<<20)) {
            var matrix=new ModelOptMatrix(index,reader,prefix);
            var raw=(List<?>)request.get("input");if(raw.size()!=Math.multiplyExact(batch,matrix.columns())) throw new IllegalArgumentException("Invalid probe input");
            float[] input=new float[raw.size()];for(int i=0;i<input.length;i++) input[i]=FloatFormats.roundBf16(((Number)raw.get(i)).floatValue());
            var weights=matrix.weights();var scales=matrix.blockScales();
            byte[] before=window(weights.file(),weights.offset()-32),tail=window(weights.file(),weights.offset()+weights.bytes()-64);
            byte[] scaleBefore=window(scales.file(),scales.offset()-32),scaleTail=window(scales.file(),scales.offset()+scales.bytes()-64);
            float[] output=new float[batch*matrix.rows()],replay=new float[output.length];
            executor.multiplyValues(matrix,input,scale,output,batch);executor.multiplyValues(matrix,input,scale,replay,batch);
            if(!Arrays.equals(output,replay)) throw new AssertionError("NVFP4 GEMM replay differs");
            var other=new FutureTask<float[]>(()-> {float[] result=new float[output.length];executor.multiplyValues(matrix,input,scale,result,batch);return result;});
            Thread.ofPlatform().start(other);if(!Arrays.equals(output,other.get())) throw new AssertionError("NVFP4 caller-thread result differs");
            executor.multiplyValues(matrix,input,scale,replay,batch);if(!Arrays.equals(output,replay)) throw new AssertionError("NVFP4 returning-caller result differs");
            var encodedInput=executor.readPacked(matrix,batch);var inputs=new Nvfp4Activation.Packed[batch];
            for(int row=0;row<batch;row++) inputs[row]=new Nvfp4Activation.Packed(Arrays.copyOfRange(encodedInput[0],row*matrix.columns()/2,(row+1)*matrix.columns()/2),
                    Arrays.copyOfRange(encodedInput[1],row*matrix.columns()/16,(row+1)*matrix.columns()/16),scale);
            executor.multiplyBatch(matrix,inputs,replay);if(!Arrays.equals(output,replay)) throw new AssertionError("NVFP4 raw/packed switch differs");
            for(int row=0;row<batch;row++) inputs[row]=new Nvfp4Activation.Packed(inputs[row].values(),inputs[row].scales(),scale*.5f);
            executor.multiplyBatch(matrix,inputs,replay);
            for(int i=0;i<output.length;i++) if(replay[i]!=FloatFormats.roundBf16(output[i]*.5f)) throw new AssertionError("NVFP4 alpha update differs");
            executor.multiplyValues(matrix,input,scale,replay,batch);if(!Arrays.equals(output,replay)) throw new AssertionError("NVFP4 raw input replay differs");
            if(!Arrays.equals(before,window(weights.file(),weights.offset()-32)) || !Arrays.equals(tail,window(weights.file(),weights.offset()+weights.bytes()-64))
                    || !Arrays.equals(scaleBefore,window(scales.file(),scales.offset()-32)) || !Arrays.equals(scaleTail,window(scales.file(),scales.offset()+scales.bytes()-64)))
                throw new AssertionError("NVFP4 GEMM changed checkpoint bytes");
            var stats=executor.statistics();executor.close();var nativeStats=FlashNextNvfp4Library.statistics();
            if(executor.statistics().residentBytes()!=0 || nativeStats.contexts()!=0 || nativeStats.plans()!=0 || nativeStats.calls()<4) throw new AssertionError("NVFP4 native resource lifecycle differs");
            var report=new LinkedHashMap<String,Object>();var encoded=new ArrayList<Double>();for(float value:output) encoded.add((double)value);
            report.put("output",encoded);report.put("matrix",prefix);report.put("batch_rows",batch);report.put("cache_replay_exact",true);report.put("caller_thread_exact",true);
            report.put("packed_values_hex",HexFormat.of().formatHex(encodedInput[0]));report.put("packed_scales_hex",HexFormat.of().formatHex(encodedInput[1]));
            report.put("raw_packed_switch_exact",true);report.put("alpha_update_exact",true);report.put("gpu_activation_quantization",true);
            report.put("source_bytes_unchanged",true);report.put("resident_bytes_after_close",0);report.put("native_contexts_after_close",nativeStats.contexts());
            report.put("native_plans_after_close",nativeStats.plans());report.put("native_calls",nativeStats.calls());report.put("native_workspace_bytes_per_plan",FlashNextNvfp4Library.WORKSPACE_BYTES);
            report.put("resident_bytes",stats.residentBytes());report.put("uploads",stats.uploads());report.put("executions",stats.executions());report.put("kernel_ns",stats.kernelNanos());
            System.out.println(Json.write(report));
        }
    }
    private static byte[] window(Path path,long offset) throws Exception {
        var bytes=ByteBuffer.allocate(64);try(var file=FileChannel.open(path,StandardOpenOption.READ)) {
            while(bytes.hasRemaining()) {int count=file.read(bytes,offset+bytes.position());if(count<=0) throw new IllegalStateException("Truncated checkpoint window");}
        }return bytes.array();
    }
}

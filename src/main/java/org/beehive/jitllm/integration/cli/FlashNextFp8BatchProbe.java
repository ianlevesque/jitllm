package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.tornado.FlashNextFp8MatrixExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextFp8Library;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.server.Json;

/** Real MTP block-FP8 tensor-core projection, encoding and caller-thread replay. */
public final class FlashNextFp8BatchProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextFp8BatchProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));String prefix=(String)request.get("matrix");
        int rows=((Number)request.get("batch_rows")).intValue(),device=((Number)request.getOrDefault("device",1)).intValue();
        float route=((Number)request.getOrDefault("route_weight",1)).floatValue();
        var index=SafeTensorIndex.open(Path.of(args[0]),name->name.startsWith(prefix+"."));
        try(var reader=new PagedTensorReader(1048576,4096);var executor=new FlashNextFp8MatrixExecutor(device,256L<<20)) {
            var matrix=new ModelOptMatrix(index,reader,prefix);var raw=(List<?>)request.get("input");
            float[] values=new float[raw.size()];for(int i=0;i<values.length;i++) values[i]=((Number)raw.get(i)).floatValue();
            float[] output=new float[rows*matrix.rows()],replay=new float[output.length];
            executor.multiply(matrix,values,output,rows,route);executor.multiply(matrix,values,replay,rows,route);
            if(!Arrays.equals(output,replay)) throw new AssertionError("FP8 replay differs");
            var future=new FutureTask<float[]>(()-> {float[] result=new float[output.length];executor.multiply(matrix,values,result,rows,route);return result;});
            Thread.ofPlatform().start(future);if(!Arrays.equals(output,future.get())) throw new AssertionError("FP8 caller-thread replay differs");
            var encoded=executor.readEncoded(matrix,rows);var scales=executor.readScales(matrix,rows);var stats=executor.statistics();executor.close();
            var nativeStats=FlashNextFp8Library.statistics();
            if(executor.statistics().residentBytes()!=0 || nativeStats.contexts()!=0 || nativeStats.plans()!=0 || nativeStats.calls()==0 || stats.kernelNanos()<=0)
                throw new AssertionError("FP8 native resources or execution differ");
            var report=new LinkedHashMap<String,Object>();report.put("matrix",prefix);report.put("batch_rows",rows);report.put("route_weight",(double)route);
            var outputValues=new ArrayList<Double>();for(float value:output) outputValues.add((double)value);
            var scaleValues=new ArrayList<Double>();for(float value:scales) scaleValues.add((double)value);
            report.put("output",outputValues);report.put("encoded_hex",HexFormat.of().formatHex(encoded));report.put("input_scales",scaleValues);
            report.put("caller_thread_exact",true);report.put("replay_exact",true);report.put("native_calls",nativeStats.calls());report.put("kernel_ns",stats.kernelNanos());
            report.put("resident_bytes",stats.residentBytes());report.put("resident_bytes_after_close",0);report.put("native_contexts_after_close",nativeStats.contexts());report.put("native_plans_after_close",nativeStats.plans());
            System.out.println(Json.write(report));
        }
    }
}

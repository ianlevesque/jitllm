package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.tornado.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.server.Json;

/** Real HC tensor-core FP32 projection with replay, thread changes and resource accounting. */
public final class FlashNextHcProjectionProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextHcProjectionProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));String name=(String)request.get("matrix");
        int device=((Number)request.getOrDefault("device",1)).intValue(),partition=((Number)request.get("partition")).intValue();
        var index=SafeTensorIndex.open(Path.of(args[0]),n->n.startsWith(name+"."));
        try(var reader=new PagedTensorReader(1048576,4096);var executor=new FlashNextBf16FloatMatrixExecutor(device,16L<<20)) {
            var matrix=new ModelOptMatrix(index,reader,name);var raw=(List<?>)request.get("input");float[] input=new float[raw.size()];for(int i=0;i<input.length;i++) input[i]=((Number)raw.get(i)).floatValue();
            float[] output=new float[matrix.rows()],replay=new float[matrix.rows()];executor.multiply(matrix,input,output,partition);executor.multiply(matrix,input,replay,partition);
            if(!Arrays.equals(output,replay)) throw new AssertionError("HC projection replay differs");
            var future=new FutureTask<float[]>(()-> {float[] result=new float[matrix.rows()];executor.multiply(matrix,input,result,partition);return result;});Thread.ofPlatform().start(future);
            if(!Arrays.equals(output,future.get())) throw new AssertionError("HC projection caller thread differs");
            var stats=executor.statistics();executor.close();var nativeStats=FlashNextBf16FloatLibrary.statistics();
            if(stats.peakBytes()>16L<<20 || stats.kernelNanos()<=0 || nativeStats.calls()==0 || nativeStats.contexts()!=0 || nativeStats.plans()!=0 || executor.statistics().residentBytes()!=0)
                throw new AssertionError("HC projection execution/resources differ");
            var report=new LinkedHashMap<String,Object>();var result=new ArrayList<Double>();for(float value:output) result.add((double)value);
            report.put("matrix",name);report.put("partition",partition);report.put("output",result);report.put("replay_exact",true);report.put("caller_thread_exact",true);
            report.put("peak_bytes",stats.peakBytes());report.put("native_calls",nativeStats.calls());report.put("kernel_ns",stats.kernelNanos());report.put("resources_after_close",0);
            System.out.println(Json.write(report));
        }
    }
}

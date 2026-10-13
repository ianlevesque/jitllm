package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.cpu.Qwen4Moe;
import org.beehive.jitllm.backend.tornado.FlashNextFp8MoeExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextFp8Library;
import org.beehive.jitllm.backend.tornado.FlashNextSharedBf16Library;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

/** Real MTP FP8 MoE under a cache budget that forces eviction, with exact replay. */
public final class FlashNextFp8MoeProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextFp8MoeProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));String prefix="mtp.layers.0.mlp";
        var config=Qwen4Configuration.load(Path.of(args[0]),-1);var index=SafeTensorIndex.open(Path.of(args[0]),name->name.startsWith(prefix+"."));
        long budget=((Number)request.getOrDefault("gpu_cache_bytes",128L<<20)).longValue();
        try(var reader=new PagedTensorReader(1048576,4096);var executor=new FlashNextFp8MoeExecutor(((Number)request.getOrDefault("device",1)).intValue(),budget)) {
            var outputs=new ArrayList<Object>();
            try(var layer=new Qwen4Moe(config,index,reader,prefix,null,executor)) {
                for(var row:(List<?>)request.get("inputs")) {
                    var raw=(List<?>)row;float[] values=new float[raw.size()];for(int i=0;i<values.length;i++) values[i]=FloatFormats.roundBf16(((Number)raw.get(i)).floatValue());
                    var output=layer.forward(values);
                    var replayTask=new FutureTask<Qwen4Moe.Result>(() -> layer.forward(values));Thread.ofPlatform().start(replayTask);var replay=replayTask.get();
                    if(!Arrays.equals(output.output(),replay.output()) || !Arrays.equals(output.routing().experts(),replay.routing().experts())
                            || !Arrays.equals(output.routing().weights(),replay.routing().weights())) throw new AssertionError("FP8 MoE replay differs");
                    var result=new ArrayList<Double>();for(float value:output.output()) result.add((double)value);
                    var ids=new ArrayList<Integer>();for(int value:output.routing().experts()) ids.add(value);
                    var weights=new ArrayList<Double>();for(float value:output.routing().weights()) weights.add((double)value);
                    var item=new LinkedHashMap<String,Object>();item.put("output",result);item.put("experts",ids);item.put("weights",weights);
                    if(Boolean.TRUE.equals(request.get("trace"))) {
                        var trace=new LinkedHashMap<String,Object>();for(var entry:layer.diagnosticIntermediates().entrySet()) {
                            var list=new ArrayList<Double>();for(float value:entry.getValue()) list.add((double)value);trace.put(entry.getKey(),list);
                        }item.put("trace",trace);
                    }
                    outputs.add(item);
                }
            }
            var s=executor.statistics();executor.close();var nativeStats=FlashNextFp8Library.statistics();
            if(FlashNextSharedBf16Library.statistics().contexts()!=0 || executor.statistics().residentBytes()!=0 || nativeStats.contexts()!=0 || nativeStats.plans()!=0 || FlashNextFp8Library.registeredWeights()!=0)
                throw new AssertionError("FP8 MoE resources leaked");
            var report=new LinkedHashMap<String,Object>();report.put("outputs",outputs);report.put("budget_bytes",budget);report.put("peak_bytes",s.peakBytes());
            report.put("evictions",s.evictions());report.put("uploads",s.uploads());report.put("forwards",s.forwards());report.put("layer_builds",s.layerBuilds());
            report.put("route_download_bytes",s.idDownloadBytes());report.put("intermediate_download_bytes",s.intermediateDownloadBytes());report.put("expert_file_bytes",s.expertFileBytes());
            report.put("kernel_ns",s.kernelNanos());report.put("native_calls",nativeStats.calls());report.put("resident_bytes_after_close",0);
            report.put("native_contexts_after_close",nativeStats.contexts());report.put("native_plans_after_close",nativeStats.plans());report.put("expert_registrations_after_close",FlashNextFp8Library.registeredWeights());
            report.put("native_shared_bf16_calls",FlashNextSharedBf16Library.statistics().calls());report.put("native_shared_bf16_contexts_after_close",FlashNextSharedBf16Library.statistics().contexts());report.put("caller_thread_replay_exact",true);
            System.out.println(Json.write(report));
        }
    }
}

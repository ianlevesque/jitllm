package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.cpu.Qwen4CpuLayers;
import org.beehive.jitllm.backend.tornado.FlashNextMtpFusionExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextRmsNormLibrary;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

public final class FlashNextMtpFusionProbe {
    private static float[] values(Object raw) {var list=(List<?>)raw;float[] out=new float[list.size()];for(int i=0;i<out.length;i++) out[i]=((Number)list.get(i)).floatValue();return out;}
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextMtpFusionProbe MODEL_DIR REQUEST.json");
        var directory=Path.of(args[0]);var request=Json.parseObject(Files.readString(Path.of(args[1])));var config=Qwen4Configuration.load(directory,4096).mtpConfiguration();
        int rows=((Number)request.get("rows")).intValue(),device=((Number)request.getOrDefault("device",1)).intValue();
        var index=SafeTensorIndex.open(directory,name->name.startsWith("mtp."));
        try(var reader=new PagedTensorReader(1048576,4096);var factory=new FlashNextMtpFusionExecutor(device,128L<<20)) {
            try(var fusion=factory.newFusion(new ModelOptMatrix(index,reader,"mtp.fc_embedding"),new ModelOptMatrix(index,reader,"mtp.fc_hidden"),config.hcCount(),config.rmsNormEps(),
                    Qwen4CpuLayers.vector(index,reader,"mtp.pre_fc_norm_embedding.weight",config.dim()),Qwen4CpuLayers.vector(index,reader,"mtp.pre_fc_norm_hidden.weight",config.hyperDim()))) {
                var embeddings=values(request.get("embeddings"));var hidden=values(request.get("hidden"));var output=fusion.forward(embeddings,hidden,rows);
                var stages=new LinkedHashMap<String,Object>();if(Boolean.TRUE.equals(request.get("trace"))) for(var stage:factory.intermediates(fusion,rows).entrySet()) {
                    var data=new ArrayList<Double>();for(float value:stage.getValue()) data.add((double)value);stages.put(stage.getKey(),data);
                }
                if(!Arrays.equals(output,fusion.forward(embeddings,hidden,rows))) throw new AssertionError("MTP fusion replay differs");
                var future=new FutureTask<float[]>(()->fusion.forward(embeddings,hidden,rows));Thread.ofPlatform().start(future);
                if(!Arrays.equals(output,future.get())) throw new AssertionError("MTP fusion caller thread differs");
                var stats=factory.statistics();fusion.close();factory.close();
                if(factory.statistics().residentBytes()!=0 || FlashNextRmsNormLibrary.statistics().contexts()!=0 || stats.peakBytes()>128L<<20 || stats.kernelNanos()<=0) throw new AssertionError("MTP fusion resources differ");
                var report=new LinkedHashMap<String,Object>();var values=new ArrayList<Double>();for(float value:output) values.add((double)value);
                report.put("output",values);report.put("rows",rows);report.put("replay_exact",true);report.put("caller_thread_exact",true);
                if(!stages.isEmpty()) report.put("stages",stages);
                report.put("peak_bytes",stats.peakBytes());report.put("builds",stats.builds());report.put("evictions",stats.evictions());report.put("kernel_ns",stats.kernelNanos());report.put("native_norm_calls",FlashNextRmsNormLibrary.statistics().calls());report.put("native_norm_contexts_after_close",FlashNextRmsNormLibrary.statistics().contexts());report.put("resident_bytes_after_close",factory.statistics().residentBytes());
                System.out.println(Json.write(report));
            }
        }
    }
}

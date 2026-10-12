package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextLayerExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor.GroupedShape;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

/** Actual CUDA HC/PLE, bounded table paging and nested device checkpoint checks. */
public final class FlashNextHcPleProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextHcPleProbe MODEL_DIR REQUEST.json");
        Path model=Path.of(args[0]);var config=Qwen4Configuration.load(model,-1);
        var request=Json.parseObject(Files.readString(Path.of(args[1])));
        int device=((Number)request.getOrDefault("device",1)).intValue();boolean projected=Boolean.TRUE.equals(request.get("gpu_projections"));
        try(var matrices=projected ? new FlashNextMatrixExecutor(device,256L<<20) : null;
            var executor=new FlashNextLayerExecutor(device,((Number)request.getOrDefault("checkpoint_capacity",4)).intValue(),16L<<20)) {
            try(var tiny=new FlashNextLayerExecutor(device,4,64)) {
                try {tiny.newHyperConnection(new GroupedShape(config.dim(),config.hcCount(),config.rmsNormEps()),config.hcLowrank(),new float[config.hyperDim()]);
                    throw new AssertionError("HC exceeded its buffer budget");} catch(IllegalStateException expected) {}
                if(tiny.statistics().residentBytes()!=0) throw new AssertionError("Rejected GPU HC leaked buffers");
            }
            var report=FlashNextLayerProbe.hyperPle(model,config,request,matrices,executor);
            if(matrices!=null) {
                var projectionStats=matrices.statistics();
                report.put("bf16_gemm_executions",projectionStats.bf16Executions());report.put("bf16_gemm_cache_bytes",projectionStats.bf16ResidentBytes());
                matrices.close();
                if(matrices.statistics().residentBytes()!=0 || matrices.statistics().bf16ResidentBytes()!=0) throw new AssertionError("Projection cache leaked after close");
                report.put("projection_bytes_after_close",0);
            }
            var stats=executor.statistics();
            if(stats.residentBytes()!=0 || stats.hyperConnections()!=0 || stats.pleSessions()!=0 || stats.liveSnapshots()!=0) throw new AssertionError("GPU HC/PLE resources leaked");
            report.put("gpu_hc_ple",Map.of("hyper_mixes",stats.hyperMixes(),"hyper_combines",stats.hyperCombines(),"ple_forwards",stats.pleForwards(),
                    "checkpoints",stats.checkpoints(),"restores",stats.restores(),"resets",stats.resets(),"kernel_ns",stats.kernelNanos(),
                    "state_host_download_bytes",stats.stateHostDownloadBytes(),"resident_bytes_after_close",stats.residentBytes(),"buffer_budget_refused",true));
            report.put("gpu_projections",projected);System.out.println(Json.write(report));
        }
    }
}

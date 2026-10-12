package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextQsaExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

/** Actual GPU QSA with physical FP8 KV pages and fresh BF16 prefill. */
public final class FlashNextQsaProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextQsaProbe MODEL_DIR REQUEST.json");
        var model=Path.of(args[0]);var request=Json.parseObject(Files.readString(Path.of(args[1])));var config=Qwen4Configuration.load(model,-1);
        int device=((Number)request.getOrDefault("device",1)).intValue();boolean projected=Boolean.TRUE.equals(request.get("gpu_projections"));
        try(var matrices=projected ? new FlashNextMatrixExecutor(device,256L<<20) : null;
            var executor=new FlashNextQsaExecutor(device,4,1L<<30)) {
            var report=FlashNextLayerProbe.qsaResult(model,config,request,matrices,executor);var stats=executor.statistics();
            if(stats.reservedBytes()!=0 || stats.sessions()!=0 || stats.liveSnapshots()!=0) throw new AssertionError("QSA resources leaked");
            report.put("gpu_qsa",Map.of("forwards",stats.forwards(),"restores",stats.restores(),"kernel_ns",stats.kernelNanos(),"kv_host_download_bytes",stats.kvHostDownloadBytes(),
                    "index_host_download_bytes",stats.indexHostDownloadBytes(),"reserved_bytes_after_close",stats.reservedBytes()));
            report.put("gpu_projections",projected);System.out.println(Json.write(report));
        }
    }
}

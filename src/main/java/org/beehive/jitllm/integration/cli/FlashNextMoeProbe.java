package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4Library;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

/** Real MoE with packed native GEMMs. Routing, activation packing and gates remain host operations. */
public final class FlashNextMoeProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextMoeProbe MODEL_DIR REQUEST.json");
        var model=Path.of(args[0]);var request=Json.parseObject(Files.readString(Path.of(args[1])));
        int device=((Number)request.getOrDefault("device",1)).intValue();boolean nativePacked=Boolean.TRUE.equals(request.get("gpu_nvfp4"));
        try(var matrices=new FlashNextMatrixExecutor(device,256L<<20,nativePacked)) {
            var report=FlashNextLayerProbe.moeResult(model,Qwen4Configuration.load(model,-1),request,matrices);var stats=matrices.statistics();
            if(nativePacked && stats.nvfp4Executions()==0) throw new AssertionError("MoE did not execute native NVFP4");
            matrices.close();var nativeStats=FlashNextNvfp4Library.statistics();
            if(matrices.statistics().residentBytes()!=0 || nativeStats.contexts()!=0 || nativeStats.plans()!=0) throw new AssertionError("MoE projection resources leaked");
            report.put("gpu_projections",Map.of("executions",stats.executions(),"nvfp4_executions",stats.nvfp4Executions(),"native_calls",nativeStats.calls(),
                    "kernel_ns",stats.kernelNanos(),"resident_bytes_after_close",matrices.statistics().residentBytes(),"native_contexts_after_close",nativeStats.contexts()));
            System.out.println(Json.write(report));
        }
    }
}

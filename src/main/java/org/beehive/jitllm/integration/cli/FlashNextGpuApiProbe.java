package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.beehive.jitllm.api.*;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4Library;
import org.beehive.jitllm.backend.tornado.FlashNextBf16FloatLibrary;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.server.Json;

/** Normal CUDA API generation, simultaneous session callers, reset/replay and resource release. */
public final class FlashNextGpuApiProbe {
    private static long descriptors(Set<String> checkpointFiles) throws Exception {
        try(var files=Files.list(Path.of("/proc/self/fd"))) {
            return files.filter(file -> {
                try {return checkpointFiles.contains(Files.readSymbolicLink(file).toString());}
                catch(Exception closed) {return false;}
            }).count();
        }
    }
    private static Map<String,Object> generate(GenerationSession session,Map<String,Object> request,List<Integer> expected) {
        var ids=new ArrayList<Integer>();var text=new StringBuilder();
        var output=session.generate(GenerationRequest.builder().prompt((String)request.get("prompt"))
                .temperature(0).maxNewTokens(((Number)request.getOrDefault("max_tokens",3)).intValue())
                .onEvent(event->{ids.add(event.tokenId());text.append(event.text());}).build());
        System.err.println("API generated "+ids+" at position "+session.position());
        if(!ids.equals(expected) || !text.toString().equals(output.text()) || output.generatedTokens()!=ids.size()
                || session.position()!=output.promptTokens()+output.generatedTokens()) throw new AssertionError("API output/stream/position mismatch: "+ids);
        if(request.containsKey("expected_prompt_tokens") && output.promptTokens()!=((Number)request.get("expected_prompt_tokens")).intValue())
            throw new AssertionError("API prompt budget differs from the checkpoint template");
        return Map.of("generated_tokens",ids,"text",output.text(),"prompt_tokens",output.promptTokens(),
                "generated_count",output.generatedTokens(),"position",session.position(),"finish_reason",output.finishReason().name());
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextGpuApiProbe MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));
        System.setProperty("jitllm.qwen4.device",String.valueOf(((Number)request.getOrDefault("device",1)).intValue()));
        Path checkpoint=Path.of(args[0]).toRealPath();var checkpointFiles=new HashSet<String>();
        try(var files=Files.list(checkpoint)) {
            for(var file:files.filter(path->path.toString().endsWith(".safetensors")).toList()) checkpointFiles.add(file.toRealPath().toString());
        }
        int context=((Number)request.getOrDefault("context_length",4096)).intValue();
        var expected=((List<?>)request.get("expected_tokens")).stream().map(v->((Number)v).intValue()).toList();
        long before=descriptors(checkpointFiles),start=System.nanoTime();
        try {
            try(var rejected=LocalModels.load(checkpoint,ModelOptions.builder().backend(BackendId.CUDA).contextLength(context).storageOptions(StorageOptions.fp16()).build())) {
                throw new AssertionError("FP16 request silently used FP8");
            }
        } catch(UnsupportedOperationException expectedRefusal) {
            if(!expectedRefusal.getMessage().contains("FP8")) throw expectedRefusal;
        }
        if(descriptors(checkpointFiles)!=before || FlashNextNvfp4Library.registeredWeights()!=0 || FlashNextNvfp4Library.statistics().contexts()!=0)
            throw new AssertionError("Rejected API load leaked resources");
        var model=LocalModels.load(checkpoint,ModelOptions.builder().backend(BackendId.CUDA).contextLength(context)
                .maxConcurrentSessions(3).thinkingMode(ThinkingMode.DISABLED).build());
        var generator=(TextGenerationModel)model;var sessions=new ArrayList<GenerationSession>();
        var results=new ArrayList<Map<String,Object>>();Object prepared=null;long during=0;
        try {
            for(int i=0;i<3;i++) {
                var session=generator.newSession();sessions.add(session);var info=session.prepare();prepared=info.toString();
                if(!"CUDA".equals(info.backend()) || !info.kvCache().startsWith("FP8") || session.position()!=0 || !info.equals(session.prepare()))
                    throw new AssertionError("prepare() changed state or described a different backend");
            }
            try {generator.newSession();throw new AssertionError("Fourth session admitted");} catch(IllegalStateException expectedRefusal) {}
            try {model.close();throw new AssertionError("Model closed with live sessions");} catch(IllegalStateException expectedRefusal) {}
            try(var workers=Executors.newFixedThreadPool(3)) {
                var futures=new ArrayList<Future<Map<String,Object>>>();
                for(var session:sessions) futures.add(workers.submit(()->generate(session,request,expected)));
                for(var future:futures) results.add(future.get());
            }
            sessions.getFirst().reset();if(sessions.getFirst().position()!=0) throw new AssertionError("Reset advanced position");
            var replay=generate(sessions.getFirst(),request,expected);
            if(!replay.equals(results.getFirst())) throw new AssertionError("Reset/replay differs");
            results.add(replay);during=descriptors(checkpointFiles);
        } finally {for(var session:sessions) session.close();model.close();}
        model.close();long after=descriptors(checkpointFiles);var nativeStats=FlashNextNvfp4Library.statistics();
        var floatStats=FlashNextBf16FloatLibrary.statistics();
        if(after!=before || nativeStats.contexts()!=0 || nativeStats.plans()!=0 || FlashNextNvfp4Library.registeredWeights()!=0 || nativeStats.calls()==0)
            throw new AssertionError("CUDA API native/checkpoint resources leaked or did not execute");
        if(floatStats.contexts()!=0 || floatStats.plans()!=0 || floatStats.calls()==0)
            throw new AssertionError("CUDA API HC/QSA FP32 GEMMs leaked resources or did not execute");
        var report=new LinkedHashMap<String,Object>();report.put("passed",true);report.put("execution",prepared);report.put("context_length",context);
        report.put("concurrent_callers",3);report.put("device_execution_serialized",true);report.put("fourth_session_refused",true);
        report.put("close_with_live_sessions_refused",true);report.put("explicit_fp16_refused",true);report.put("reset_replay_exact",true);report.put("results",results);
        report.put("checkpoint_descriptors_before",before);report.put("checkpoint_descriptors_during",during);report.put("checkpoint_descriptors_after",after);
        report.put("native_gemms",nativeStats.calls());report.put("native_contexts_after_close",nativeStats.contexts());report.put("native_plans_after_close",nativeStats.plans());
        report.put("expert_registrations_after_close",FlashNextNvfp4Library.registeredWeights());report.put("elapsed_seconds",(System.nanoTime()-start)/1e9);
        report.put("bf16_float_gemms",floatStats.calls());report.put("bf16_float_contexts_after_close",floatStats.contexts());report.put("bf16_float_plans_after_close",floatStats.plans());
        report.put("scope","Three API callers serialized on shared device workspaces, 4096-token capacity and one short prompt/reset. Full-context concurrent capacity, batching, MTP, vision and broader chat replay remain unqualified.");
        System.out.println(Json.write(report));
    }
}

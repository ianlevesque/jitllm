package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.Qwen4CpuEngine;
import org.beehive.jitllm.server.Json;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextDeltaNetExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextLayerExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextQsaExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4Library;

/** Native full-model reference runner; this host path precedes the device serving integration. */
public final class FlashNextCpuGenerate {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextCpuGenerate MODEL_DIR REQUEST.json");
        var request=Json.parseObject(Files.readString(Path.of(args[1])));
        int maximum=((Number)request.getOrDefault("max_tokens",1)).intValue();
        long start=System.nanoTime();
        boolean projected=Boolean.TRUE.equals(request.get("gpu_projections"));
        boolean recurrent=Boolean.TRUE.equals(request.get("gpu_recurrence"));
        boolean layerOps=Boolean.TRUE.equals(request.get("gpu_hc_ple"));
        boolean sparse=Boolean.TRUE.equals(request.get("gpu_qsa"));
        boolean packed=Boolean.TRUE.equals(request.get("gpu_nvfp4"));
        if(packed && !projected) throw new IllegalArgumentException("Native NVFP4 requires GPU projections");
        try(var executor=projected ? new FlashNextMatrixExecutor(((Number)request.getOrDefault("device",1)).intValue(),((Number)request.getOrDefault("gpu_cache_bytes",16L<<30)).longValue(),packed) : null;
            var recurrence=recurrent ? new FlashNextDeltaNetExecutor(((Number)request.getOrDefault("device",1)).intValue(),
                    ((Number)request.getOrDefault("checkpoint_capacity",8)).intValue(),((Number)request.getOrDefault("gpu_state_bytes",4L<<30)).longValue()) : null;
            var layers=layerOps ? new FlashNextLayerExecutor(((Number)request.getOrDefault("device",1)).intValue(),
                    ((Number)request.getOrDefault("checkpoint_capacity",8)).intValue(),((Number)request.getOrDefault("gpu_layer_bytes",64L<<20)).longValue()) : null;
            var attention=sparse ? new FlashNextQsaExecutor(((Number)request.getOrDefault("device",1)).intValue(),
                    ((Number)request.getOrDefault("checkpoint_capacity",8)).intValue(),((Number)request.getOrDefault("gpu_qsa_bytes",4L<<30)).longValue()) : null;
            var engine=new Qwen4CpuEngine(Path.of(args[0]),-1,1048576,(loaded,total)->System.err.println("Loaded layer "+loaded+"/"+total),executor,recurrence,layers,attention)) {
            List<Integer> tokens;
            if(request.containsKey("input_tokens")) tokens=((List<?>)request.get("input_tokens")).stream().map(v->((Number)v).intValue()).toList();
            else tokens=engine.tokenizer().encode((String)request.get("prompt"),engine.tokenizer().getSpecialTokens().keySet());
            if(tokens.isEmpty() || maximum<1) throw new IllegalArgumentException("Nonempty prompt and positive token count are required");
            var sequence=engine.newSequence();Qwen4CpuEngine.Output output=null;
            sequence.beginInitialPrefill(tokens.size());
            boolean rollbackProbe=Boolean.TRUE.equals(request.get("rollback_probe"));
            if(rollbackProbe) {
                int failureLayer=((Number)request.getOrDefault("rollback_layer",20)).intValue();
                if(failureLayer<0 || failureLayer>=engine.configuration().numberOfLayers()) throw new IllegalArgumentException("Invalid rollback probe layer");
                var injected=new IllegalStateException("Injected layer failure for sequence rollback validation");
                try {
                    engine.forward(tokens.getFirst(),sequence,false,(layer,values)-> {if(layer==failureLayer) throw injected;});
                    throw new AssertionError("Rollback probe failed to inject its layer error");
                } catch(IllegalStateException failure) {if(failure!=injected || failure.getSuppressed().length!=0) throw failure;}
                if(sequence.position()!=0 || (recurrence!=null && recurrence.statistics().liveSnapshots()!=0) || (layers!=null && layers.statistics().liveSnapshots()!=0)
                        || (attention!=null && attention.statistics().liveSnapshots()!=0))
                    throw new AssertionError("Failed token retained position/checkpoints");
            }
            List<Object> trace=new ArrayList<>();
            var callback=Boolean.TRUE.equals(request.get("trace_layers")) ? (java.util.function.BiConsumer<Integer,float[]>)(layer,values)-> {
                double squares=0,largest=0;for(float value:values) {squares+=value*value;largest=Math.max(largest,Math.abs(value));}
                trace.add(Map.of("position",sequence.position(),"layer",layer,"rms",Math.sqrt(squares/values.length),"max_abs",largest));
                System.err.println("Forward position "+sequence.position()+" layer "+layer);
            } : null;
            for(int i=0;i<tokens.size();i++) output=engine.forward(tokens.get(i),sequence,i==tokens.size()-1,callback);
            List<Integer> generated=new ArrayList<>();
            for(int i=0;i<maximum;i++) {
                int token=output.greedy();generated.add(token);
                System.err.println("Generated token "+token);
                if(token==engine.configuration().eosToken() || token==engine.tokenizer().getSpecialTokens().get("<|im_end|>")) break;
                if(i+1<maximum) output=engine.forward(token,sequence,true,callback);
            }
            var report=new LinkedHashMap<String,Object>();
            report.put("input_tokens",tokens);report.put("generated_tokens",generated);report.put("text",engine.tokenizer().decode(generated));
            report.put("elapsed_seconds",(System.nanoTime()-start)/1e9);report.put("cache_resident_bytes",engine.statistics().residentBytes());report.put("trace",trace);
            var paging=engine.statistics();
            report.put("paged_reader",Map.of("hits",paging.hits(),"misses",paging.misses(),"file_bytes_read",paging.bytesRead(),"resident_bytes",paging.residentBytes(),"budget_bytes",paging.budgetBytes()));
            if(rollbackProbe) report.put("rollback_probe",Map.of("injected_layer",((Number)request.getOrDefault("rollback_layer",20)).intValue(),"position_restored",true,"transaction_checkpoints_released",true));
            if(executor!=null) {
                var stats=executor.statistics();
                report.put("execution","GPU weight projections"+(recurrent ? " and DeltaNet" : "")+(layerOps ? " and HC/PLE" : "")+(sparse ? " and QSA" : "")
                        +"; "+(sparse ? "MoE nonlinear work remains on the host" : "QSA and MoE nonlinear work remains on the host"));
                report.put("device",executor.deviceName());report.put("gpu_cache_bytes",stats.residentBytes());
                report.put("gpu_uploads",stats.uploads());report.put("gpu_evictions",stats.evictions());report.put("gpu_executions",stats.executions());report.put("gpu_kernel_ns",stats.kernelNanos());
                report.put("bf16_gemm_cache_bytes",stats.bf16ResidentBytes());report.put("bf16_gemm_executions",stats.bf16Executions());
                report.put("nvfp4_gemm_cache_bytes",stats.nvfp4ResidentBytes());report.put("nvfp4_gemm_executions",stats.nvfp4Executions());
            }
            if(recurrence!=null) {
                var stats=recurrence.statistics();
                report.put("gpu_recurrence",Map.of("resident_bytes",stats.residentBytes(),"sessions",stats.sessions(),"live_snapshots",stats.liveSnapshots(),
                        "forwards",stats.forwards(),"checkpoints",stats.checkpoints(),"restores",stats.restores(),"kernel_ns",stats.kernelNanos(),"state_host_download_bytes",stats.stateHostDownloadBytes()));
            }
            if(layers!=null) {
                var stats=layers.statistics();
                var operations=new LinkedHashMap<String,Object>();
                operations.put("resident_bytes",stats.residentBytes());operations.put("hyper_connections",stats.hyperConnections());operations.put("ple_sessions",stats.pleSessions());
                operations.put("live_snapshots",stats.liveSnapshots());operations.put("hyper_mixes",stats.hyperMixes());operations.put("hyper_combines",stats.hyperCombines());
                operations.put("ple_forwards",stats.pleForwards());operations.put("checkpoints",stats.checkpoints());operations.put("restores",stats.restores());
                operations.put("kernel_ns",stats.kernelNanos());operations.put("state_host_download_bytes",stats.stateHostDownloadBytes());report.put("gpu_hc_ple",operations);
            }
            if(attention!=null) {
                var stats=attention.statistics();var operations=new LinkedHashMap<String,Object>();
                operations.put("reserved_bytes",stats.reservedBytes());operations.put("materialized_page_bytes",stats.materializedPageBytes());
                operations.put("sessions",stats.sessions());operations.put("pages",stats.pages());operations.put("live_snapshots",stats.liveSnapshots());
                operations.put("forwards",stats.forwards());operations.put("restores",stats.restores());operations.put("kernel_ns",stats.kernelNanos());
                operations.put("kv_host_download_bytes",stats.kvHostDownloadBytes());operations.put("index_host_download_bytes",stats.indexHostDownloadBytes());
                report.put("gpu_qsa",operations);
            }
            sequence.close();
            if(attention!=null) {
                var stats=attention.statistics();
                if(stats.reservedBytes()!=0 || stats.sessions()!=0 || stats.liveSnapshots()!=0 || stats.pages()!=0) throw new AssertionError("QSA sequence resources leaked");
                report.put("gpu_qsa_after_sequence_close_bytes",stats.reservedBytes());
            }
            if(recurrence!=null) report.put("gpu_recurrence_after_sequence_close_bytes",recurrence.statistics().residentBytes());
            if(layers!=null) {
                report.put("gpu_ple_sessions_after_sequence_close",layers.statistics().pleSessions());
                engine.close();report.put("gpu_layer_bytes_after_engine_close",layers.statistics().residentBytes());
            }
            if(packed) {
                executor.close();var nativeStats=FlashNextNvfp4Library.statistics();
                if(nativeStats.contexts()!=0 || nativeStats.plans()!=0) throw new AssertionError("NVFP4 native resources leaked");
                report.put("gpu_nvfp4_native_contexts_after_close",nativeStats.contexts());report.put("gpu_nvfp4_native_plans_after_close",nativeStats.plans());
                report.put("gpu_nvfp4_native_calls",nativeStats.calls());
            }
            System.out.println(Json.write(report));
        }
    }
}

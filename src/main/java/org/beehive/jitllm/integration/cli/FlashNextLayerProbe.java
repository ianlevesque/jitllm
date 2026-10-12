package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.Qwen4CpuLayers;
import org.beehive.jitllm.backend.cpu.Qwen4DeltaNet;
import org.beehive.jitllm.backend.cpu.Qwen4QsaAttention;
import org.beehive.jitllm.backend.cpu.Qwen4Moe;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;
import org.beehive.jitllm.inference.op.FlashNextLayerExecutor;
import org.beehive.jitllm.inference.op.QsaExecutor;

/** Real PLE/hyper-connection execution, including state rollback and sequence reset checks. */
public final class FlashNextLayerProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextLayerProbe MODEL_DIR REQUEST.json");
        Path model=Path.of(args[0]);
        Qwen4Configuration config=Qwen4Configuration.load(model,-1);
        Map<String,Object> request=Json.parseObject(Files.readString(Path.of(args[1])));
        if("gdn".equals(request.get("mode"))) { deltaNet(model,config,request); return; }
        if("qsa".equals(request.get("mode"))) { qsa(model,config,request); return; }
        if("moe".equals(request.get("mode"))) { moe(model,config,request); return; }
        if("hybrid0".equals(request.get("mode"))) { hybrid0(model,config,request); return; }
        System.out.println(Json.write(hyperPle(model,config,request,null,null)));
    }
    public static Map<String,Object> hyperPle(Path model,Qwen4Configuration config,Map<String,Object> request,MatrixExecutor matrices,FlashNextLayerExecutor executor) throws Exception {
        String prefix="model.language_model.layers.1";
        SafeTensorIndex index=SafeTensorIndex.open(model,name -> name.startsWith(prefix+".ple.") || name.startsWith(prefix+".attn_hyper_connection."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var ple=new Qwen4CpuLayers.Ple(config,index,reader,prefix+".ple",matrices,executor);
            try(var hyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".attn_hyper_connection",true,matrices,executor);
                var state=ple.newState()) {
            List<Object> results=new ArrayList<>();
            float[] hidden=values((List<?>)request.get("hidden"));
            float[] block=values((List<?>)request.get("block"));
            int rows=((Number)request.getOrDefault("hc_rows",1)).intValue();
            boolean fused=Boolean.TRUE.equals(request.get("fused_ple_decode"));
            var hyperTrace=new LinkedHashMap<String,Object>();
            Qwen4CpuLayers.HyperConnection.Mixed mixed=hyper.mix(hidden,rows,Boolean.TRUE.equals(request.get("trace_hyper")) ?
                    (name,values)->hyperTrace.put(name,floats(values)) : null);
            float[] combined=hyper.combine(block,mixed);
            if(executor!=null) {
                var replayMix=hyper.mix(hidden,rows);
                if(!Arrays.equals(mixed.output(),replayMix.output()) || !Arrays.equals(combined,hyper.combine(block,replayMix)))
                    throw new AssertionError("GPU HC graph/GEMM reuse mismatch");
            }
            for(Object value:(List<?>)request.get("tokens")) {
                int token=((Number)value).intValue();
                try(var snapshot=state.snapshot()) {
                    float[] output=ple.forward(token,hidden,state,fused);
                    try(var accepted=state.snapshot()) {
                        state.restore(snapshot);
                        float[] replay=ple.forward(token,hidden,state,fused);
                        if(!Arrays.equals(output,replay)) throw new AssertionError("PLE rollback/replay mismatch");
                        if(executor==null) try(var replayed=state.snapshot()) {
                            if(!Arrays.equals(accepted.convolution(),replayed.convolution())) throw new AssertionError("PLE state replay mismatch");
                        }
                        state.restore(accepted);
                    }
                    results.add(floats(output));
                }
            }
            state.reset();
            int first=((Number)((List<?>)request.get("tokens")).get(0)).intValue();
            if(!floats(ple.forward(first,hidden,state,fused)).equals(results.get(0))) throw new AssertionError("PLE reset mismatch");
            if(executor!=null) {
                try(var foreign=ple.newState();var saved=state.snapshot()) {
                    var forged=new Qwen4CpuLayers.Ple.State.Snapshot(foreign,saved.history(),new float[0],saved.device());
                    try {foreign.restore(forged);throw new AssertionError("Foreign PLE checkpoint accepted");} catch(IllegalArgumentException expected) {}
                    if(!floats(ple.forward(first,hidden,foreign,fused)).equals(results.get(0))) throw new AssertionError("PLE sequences share state");
                }
                var snapshots=new ArrayList<Qwen4CpuLayers.Ple.State.Snapshot>();
                try {
                    for(int i=0;i<((Number)request.getOrDefault("checkpoint_capacity",4)).intValue();i++) snapshots.add(state.snapshot());
                    try {state.snapshot();throw new AssertionError("Unbounded PLE checkpoint pool");} catch(IllegalStateException expected) {}
                } finally {for(var saved:snapshots) saved.close();}
                var stale=state.snapshot();state.reset();
                try(var current=state.snapshot()) {
                    try {state.restore(stale);throw new AssertionError("PLE checkpoint survived reset");} catch(IllegalArgumentException expected) {}
                    stale.close();state.restore(current);
                    var held=new ArrayList<Qwen4CpuLayers.Ple.State.Snapshot>();
                    try {
                        for(int i=1;i<((Number)request.getOrDefault("checkpoint_capacity",4)).intValue();i++) held.add(state.snapshot());
                        try {state.snapshot();throw new AssertionError("Stale PLE close released a current slot");} catch(IllegalStateException expected) {}
                    } finally {for(var saved:held) saved.close();}
                }
                state.reset();
                float[] added=ple.apply(first,hidden,state,fused),delta=values((List<?>)results.getFirst());
                for(int i=0;i<added.length;i++) if(added[i]!=FloatFormats.roundBf16(hidden[i]+delta[i])) throw new AssertionError("GPU PLE residual add differs");
            }
            var stats=reader.statistics();
            var report=new LinkedHashMap<String,Object>();
            report.put("normalized",floats(mixed.normalized()));report.put("mixed",floats(mixed.output()));report.put("combined",floats(combined));report.put("ple",results);
            if(!hyperTrace.isEmpty()) report.put("hyper_trace",hyperTrace);
            report.put("rollback_exact",true);report.put("reset_exact",true);report.put("hc_rows",rows);report.put("fused_ple_decode",fused);
            report.put("cache_resident_bytes",stats.residentBytes());report.put("page_misses",stats.misses());report.put("file_bytes_read",stats.bytesRead());
            if(executor!=null) {
                state.close();hyper.close();
                try {ple.forward(first,hidden,state,fused);throw new AssertionError("Closed PLE state executed");} catch(IllegalStateException expected) {}
                try {hyper.mix(hidden);throw new AssertionError("Closed HC executed");} catch(IllegalStateException expected) {}
                report.put("foreign_checkpoint_refused",true);report.put("reset_checkpoint_refused",true);report.put("independent_sequences",true);
                report.put("checkpoint_pool_bounded",true);report.put("use_after_close_refused",true);report.put("residual_add_exact",true);
            }
            return report;
            }
        }
    }
    private static void deltaNet(Path model,Qwen4Configuration config,Map<String,Object> request) throws Exception {
        String prefix="model.language_model.layers.0.linear_attn";
        SafeTensorIndex index=SafeTensorIndex.open(model,name -> name.startsWith(prefix+"."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var layer=new Qwen4DeltaNet(config,index,reader,prefix);
            var state=layer.newState();
            List<Object> results=new ArrayList<>();
            for(Object value:(List<?>)request.get("inputs")) {
                float[] input=values((List<?>)value);
                var before=state.snapshot();
                float[] output=layer.forward(input,state);
                var after=state.snapshot();
                state.restore(before);
                float[] replay=layer.forward(input,state);
                if(!Arrays.equals(output,replay) || !Arrays.equals(after.temporal(),state.snapshot().temporal())
                        || !Arrays.equals(after.convolution(),state.snapshot().convolution())) throw new AssertionError("DeltaNet replay mismatch");
                results.add(floats(output));
            }
            state.reset();
            float[] first=values((List<?>)((List<?>)request.get("inputs")).get(0));
            if(!floats(layer.forward(first,state)).equals(results.get(0))) throw new AssertionError("DeltaNet reset mismatch");
            System.out.println(Json.write(Map.of("outputs",results,"rollback_exact",true,"reset_exact",true,
                    "cache_resident_bytes",reader.statistics().residentBytes())));
        }
    }
    private static void qsa(Path model,Qwen4Configuration config,Map<String,Object> request) throws Exception {
        System.out.println(Json.write(qsaResult(model,config,request,null,null)));
    }
    public static Map<String,Object> qsaResult(Path model,Qwen4Configuration config,Map<String,Object> request,MatrixExecutor matrices,QsaExecutor executor) throws Exception {
        String prefix="model.language_model.layers.3.self_attn";
        SafeTensorIndex index=SafeTensorIndex.open(model,name->name.startsWith(prefix+"."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var layer=new Qwen4QsaAttention(config,index,reader,prefix,"fp8_e4m3".equals(request.get("kv_dtype")),Boolean.TRUE.equals(request.get("fused_precision")),matrices,executor);
            try(var state=layer.newState()) {
            List<Object> outputs=new ArrayList<>(),selections=new ArrayList<>();int step=0;
            boolean fresh=Boolean.TRUE.equals(request.get("fresh_prefill"));
            for(Object value:(List<?>)request.get("inputs")) {
                float[] input=values((List<?>)value);
                int[] positions=request.containsKey("positions") ? ((List<?>)((List<?>)request.get("positions")).get(step)).stream().mapToInt(p->((Number)p).intValue()).toArray()
                        : new int[]{step,step,step};
                try(var before=state.snapshot()) {
                    float[] output=layer.forward(input,positions,state,fresh);int[] selected=state.selectedTokens();
                    if(fresh) state.finishFreshPrefill();
                    state.restore(before);float[] replay=layer.forward(input,positions,state,fresh);
                    if(!Arrays.equals(output,replay) || !Arrays.equals(selected,state.selectedTokens())) throw new AssertionError("QSA rollback mismatch");
                    outputs.add(floats(output));selections.add(Arrays.stream(selected).boxed().toList());step++;
                }
            }
            state.reset();float[] first=values((List<?>)((List<?>)request.get("inputs")).get(0));
            int[] positions=request.containsKey("positions") ? ((List<?>)((List<?>)request.get("positions")).get(0)).stream().mapToInt(p->((Number)p).intValue()).toArray() : new int[]{0,0,0};
            if(!floats(layer.forward(first,positions,state,fresh)).equals(outputs.get(0))) throw new AssertionError("QSA reset mismatch");
            return new LinkedHashMap<>(Map.of("outputs",outputs,"selections",selections,"rollback_exact",true,"reset_exact",true,"cache_resident_bytes",reader.statistics().residentBytes()));
            }
        }
    }
    private static void moe(Path model,Qwen4Configuration config,Map<String,Object> request) throws Exception {
        System.out.println(Json.write(moeResult(model,config,request,null)));
    }
    public static Map<String,Object> moeResult(Path model,Qwen4Configuration config,Map<String,Object> request,MatrixExecutor matrices) throws Exception {
        String prefix="model.language_model.layers.0.mlp";
        SafeTensorIndex index=SafeTensorIndex.open(model,name->name.startsWith(prefix+"."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var layer=new Qwen4Moe(config,index,reader,prefix,matrices);
            List<Object> outputs=new ArrayList<>();
            for(Object value:(List<?>)request.get("inputs")) {
                var result=layer.forward(values((List<?>)value));
                outputs.add(Map.of("output",floats(result.output()),"experts",Arrays.stream(result.routing().experts()).boxed().toList(),"weights",floats(result.routing().weights())));
            }
            return new LinkedHashMap<>(Map.of("outputs",outputs,"input_scale",(double)layer.inputScale(),"intermediate_scale",(double)layer.intermediateScale(),"cache_resident_bytes",reader.statistics().residentBytes()));
        }
    }
    private static void hybrid0(Path model,Qwen4Configuration config,Map<String,Object> request) throws Exception {
        String prefix="model.language_model.layers.0";
        var index=SafeTensorIndex.open(model,name->name.startsWith(prefix+".") || name.equals("model.language_model.embed_tokens.weight"));
        try(var reader=new PagedTensorReader(1048576,4096)) {
            var embedding=new ModelOptMatrix(index,reader,"model.language_model.embed_tokens");
            float[] row=new float[config.dim()];embedding.readRow(((Number)request.get("token")).intValue(),row);
            float[] hidden=new float[config.hyperDim()];
            for(int branch=0;branch<config.hcCount();branch++) System.arraycopy(row,0,hidden,branch*row.length,row.length);
            var attnHyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".attn_hyper_connection",true);
            var mlpHyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".mlp_hyper_connection",true);
            var delta=new Qwen4DeltaNet(config,index,reader,prefix+".linear_attn");
            var mlp=new Qwen4Moe(config,index,reader,prefix+".mlp");
            var mixed=attnHyper.mix(hidden);var attention=delta.forward(mixed.output(),delta.newState());
            var combined=attnHyper.combine(attention,mixed);var mlpMixed=mlpHyper.mix(combined);var routed=mlp.forward(mlpMixed.output());
            var output=mlpHyper.combine(routed.output(),mlpMixed);
            System.out.println(Json.write(Map.of("embedding",floats(row),"attn_mix",floats(mixed.output()),"attn_norm",floats(mixed.normalized()),
                    "attn_out",floats(attention),"attn_combined",floats(combined),"mlp_mix",floats(mlpMixed.output()),"mlp_out",floats(routed.output()),
                    "output",floats(output),"experts",Arrays.stream(routed.routing().experts()).boxed().toList())));
        }
    }
    private static float[] values(List<?> values) {
        float[] result=new float[values.size()];
        for(int i=0; i<result.length; i++) result[i]=FloatFormats.roundBf16(((Number)values.get(i)).floatValue());
        return result;
    }
    private static List<Double> floats(float[] values) {
        List<Double> result=new ArrayList<>(values.length);
        for(float value:values) result.add((double)value);
        return result;
    }
}

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
        String prefix="model.language_model.layers.1";
        SafeTensorIndex index=SafeTensorIndex.open(model,name -> name.startsWith(prefix+".ple.") || name.startsWith(prefix+".attn_hyper_connection."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var ple=new Qwen4CpuLayers.Ple(config,index,reader,prefix+".ple");
            var hyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,prefix+".attn_hyper_connection",true);
            var state=ple.newState();
            List<Object> results=new ArrayList<>();
            float[] hidden=values((List<?>)request.get("hidden"));
            float[] block=values((List<?>)request.get("block"));
            Qwen4CpuLayers.HyperConnection.Mixed mixed=hyper.mix(hidden);
            float[] combined=hyper.combine(block,mixed);
            for(Object value:(List<?>)request.get("tokens")) {
                int token=((Number)value).intValue();
                var snapshot=state.snapshot();
                float[] output=ple.forward(token,hidden,state);
                var accepted=state.snapshot();
                state.restore(snapshot);
                float[] replay=ple.forward(token,hidden,state);
                if(!Arrays.equals(output,replay)) throw new AssertionError("PLE rollback/replay mismatch");
                if(!Arrays.equals(accepted.convolution(),state.snapshot().convolution())) throw new AssertionError("PLE state replay mismatch");
                results.add(floats(output));
            }
            state.reset();
            int first=((Number)((List<?>)request.get("tokens")).get(0)).intValue();
            if(!floats(ple.forward(first,hidden,state)).equals(results.get(0))) throw new AssertionError("PLE reset mismatch");
            System.out.println(Json.write(Map.of("normalized",floats(mixed.normalized()),"mixed",floats(mixed.output()),"combined",floats(combined),"ple",results,
                    "rollback_exact",true,"reset_exact",true,"cache_resident_bytes",reader.statistics().residentBytes())));
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
        String prefix="model.language_model.layers.3.self_attn";
        SafeTensorIndex index=SafeTensorIndex.open(model,name->name.startsWith(prefix+"."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var layer=new Qwen4QsaAttention(config,index,reader,prefix,"fp8_e4m3".equals(request.get("kv_dtype")),Boolean.TRUE.equals(request.get("fused_precision")));
            var state=layer.newState();
            List<Object> outputs=new ArrayList<>(),selections=new ArrayList<>();int step=0;
            boolean fresh=Boolean.TRUE.equals(request.get("fresh_prefill"));
            for(Object value:(List<?>)request.get("inputs")) {
                float[] input=values((List<?>)value);
                int[] positions=request.containsKey("positions") ? ((List<?>)((List<?>)request.get("positions")).get(step)).stream().mapToInt(p->((Number)p).intValue()).toArray()
                        : new int[]{step,step,step};
                var before=state.snapshot();float[] output=layer.forward(input,positions,state,fresh);int[] selected=state.selectedTokens();
                if(fresh) state.finishFreshPrefill();
                state.restore(before);float[] replay=layer.forward(input,positions,state,fresh);
                if(!Arrays.equals(output,replay) || !Arrays.equals(selected,state.selectedTokens())) throw new AssertionError("QSA rollback mismatch");
                outputs.add(floats(output));selections.add(Arrays.stream(selected).boxed().toList());step++;
            }
            state.reset();float[] first=values((List<?>)((List<?>)request.get("inputs")).get(0));
            int[] positions=request.containsKey("positions") ? ((List<?>)((List<?>)request.get("positions")).get(0)).stream().mapToInt(p->((Number)p).intValue()).toArray() : new int[]{0,0,0};
            if(!floats(layer.forward(first,positions,state,fresh)).equals(outputs.get(0))) throw new AssertionError("QSA reset mismatch");
            System.out.println(Json.write(Map.of("outputs",outputs,"selections",selections,"rollback_exact",true,"reset_exact",true,"cache_resident_bytes",reader.statistics().residentBytes())));
        }
    }
    private static void moe(Path model,Qwen4Configuration config,Map<String,Object> request) throws Exception {
        String prefix="model.language_model.layers.0.mlp";
        SafeTensorIndex index=SafeTensorIndex.open(model,name->name.startsWith(prefix+"."));
        try(PagedTensorReader reader=new PagedTensorReader(1048576,4096)) {
            var layer=new Qwen4Moe(config,index,reader,prefix);
            List<Object> outputs=new ArrayList<>();
            for(Object value:(List<?>)request.get("inputs")) {
                var result=layer.forward(values((List<?>)value));
                outputs.add(Map.of("output",floats(result.output()),"experts",Arrays.stream(result.routing().experts()).boxed().toList(),"weights",floats(result.routing().weights())));
            }
            System.out.println(Json.write(Map.of("outputs",outputs,"input_scale",(double)layer.inputScale(),"intermediate_scale",(double)layer.intermediateScale(),"cache_resident_bytes",reader.statistics().residentBytes())));
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

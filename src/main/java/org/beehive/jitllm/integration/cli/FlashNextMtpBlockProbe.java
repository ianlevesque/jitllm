package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.*;
import org.beehive.jitllm.backend.tornado.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;

/** Complete native MTP block with teacher extension and three frozen-index draft steps. */
public final class FlashNextMtpBlockProbe {
    private static float[] values(Object raw) {var list=(List<?>)raw;float[] out=new float[list.size()];for(int i=0;i<out.length;i++) out[i]=((Number)list.get(i)).floatValue();return out;}
    private static List<Double> json(float[] raw) {var out=new ArrayList<Double>();for(float v:raw) out.add((double)v);return out;}
    private static void equal(Qwen4CpuEngine.Output a,Qwen4CpuEngine.Output b) {
        if(!Arrays.equals(a.hidden(),b.hidden()) || !Arrays.equals(a.hyperHidden(),b.hyperHidden()) || !Arrays.equals(a.logits(),b.logits())) throw new AssertionError("MTP replay differs");
    }
    private static Map<String,Object> report(Qwen4CpuEngine.Output output,Qwen4MtpBlock.State state,Map<String,Object> trace) {
        trace.put("hidden",json(output.hidden()));trace.put("hyper_hidden",json(output.hyperHidden()));
        var selected=new ArrayList<Integer>();for(int token:state.selectedTokens()) selected.add(token);trace.put("selected_tokens",selected);
        if(output.logits()!=null) trace.put("greedy",output.greedy());return trace;
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextMtpBlockProbe MODEL_DIR REQUEST.json");
        var directory=Path.of(args[0]);var request=Json.parseObject(Files.readString(Path.of(args[1])));var config=Qwen4Configuration.load(directory,4096);
        int device=((Number)request.getOrDefault("device",1)).intValue();boolean logits=Boolean.TRUE.equals(request.get("with_logits"));
        var tokens=(List<?>)request.get("tokens");var hidden=(List<?>)request.get("target_hidden");var draftTokens=(List<?>)request.get("draft_tokens");
        if(tokens.size()!=hidden.size() || tokens.isEmpty() || draftTokens.size()!=3) throw new IllegalArgumentException("MTP probe needs teacher rows and three draft inputs");
        int vocabulary=Qwen4Tokenizer.load(directory).tokenCount();
        var index=SafeTensorIndex.open(directory,name->name.startsWith("mtp.") || name.equals("model.language_model.embed_tokens.weight") || name.equals("lm_head.weight"));
        try(var reader=new PagedTensorReader(1048576,4096);var matrices=new FlashNextMatrixExecutor(device,2L<<30);
            var layers=new FlashNextLayerExecutor(device,8,4L<<20,true);var qsa=new FlashNextQsaExecutor(device,8,64L<<20);
            var experts=new FlashNextFp8MoeExecutor(device,128L<<20);var fusion=new FlashNextMtpFusionExecutor(device,128L<<20);
            var block=new Qwen4MtpBlock(config,index,reader,new ModelOptMatrix(index,reader,"model.language_model.embed_tokens",matrices),
                    new ModelOptMatrix(index,reader,"lm_head",matrices),vocabulary,8,matrices,layers,qsa,experts,fusion);
            var state=block.newState()) {
            var teacher=new ArrayList<Map<String,Object>>();state.beginPrefill(tokens.size());Qwen4CpuEngine.Output last=null;
            for(int i=0;i<tokens.size();i++) {
                var trace=new LinkedHashMap<String,Object>();last=block.forward(((Number)tokens.get(i)).intValue(),values(hidden.get(i)),state,logits && i==tokens.size()-1,(name,data)->trace.put(name,json(data)));
                teacher.add(report(last,state,trace));
            }
            state.finishPrefill();var baseHidden=last.hyperHidden().clone();
            var drafts=new ArrayList<Map<String,Object>>();var original=new ArrayList<Qwen4CpuEngine.Output>();
            try(var base=state.snapshot()) {
                var h=baseHidden;
                for(var token:draftTokens) {
                    var trace=new LinkedHashMap<String,Object>();var result=block.forwardDraft(((Number)token).intValue(),h,state,logits,(name,data)->trace.put(name,json(data)));
                    drafts.add(report(result,state,trace));original.add(result);h=result.hyperHidden();
                }
                int position=state.position();try {block.forwardDraft(17,h,state,false,null);throw new AssertionError("Fourth MTP step accepted");} catch(IllegalArgumentException expected) {}
                if(position!=state.position()) throw new AssertionError("Refused MTP draft advanced state");
                state.restore(base);h=baseHidden;
                for(int i=0;i<3;i++) {var replay=block.forwardDraft(((Number)draftTokens.get(i)).intValue(),h,state,logits,null);equal(original.get(i),replay);h=replay.hyperHidden();}
                state.restore(base);
                try {block.forwardDraft(((Number)draftTokens.getFirst()).intValue(),baseHidden,state,false,(name,data)-> {if(name.equals("hyper_hidden")) throw new IllegalStateException("injected after MTP QSA append");});throw new AssertionError("MTP injection did not fire");}
                catch(IllegalStateException expected) {if(!expected.getMessage().equals("injected after MTP QSA append")) throw expected;}
                if(state.position()!=base.position() || state.liveSnapshots()!=1) throw new AssertionError("MTP failed transaction did not restore");
                equal(original.getFirst(),block.forwardDraft(((Number)draftTokens.getFirst()).intValue(),baseHidden,state,logits,null));state.restore(base);
                try(var other=block.newState()) {try {other.restore(base);throw new AssertionError("Foreign MTP snapshot accepted");} catch(IllegalArgumentException expected) {}}
            }
            if(state.liveSnapshots()!=0) throw new AssertionError("MTP snapshots leaked");
            state.reset();state.beginPrefill(tokens.size());
            for(int i=0;i<tokens.size();i++) {
                var replay=block.forward(((Number)tokens.get(i)).intValue(),values(hidden.get(i)),state,logits && i==tokens.size()-1,null);
                if(!json(replay.hidden()).equals(teacher.get(i).get("hidden")) || !json(replay.hyperHidden()).equals(teacher.get(i).get("hyper_hidden"))) throw new AssertionError("MTP reset differs");
            }
            var expired=state.snapshot();expired.close();try {state.restore(expired);throw new AssertionError("Disposed MTP snapshot accepted");} catch(IllegalArgumentException expected) {}
            var gpuQsa=qsa.statistics();var gpuExperts=experts.statistics();var gpuFusion=fusion.statistics();
            state.close();block.close();matrices.close();layers.close();qsa.close();experts.close();fusion.close();
            if(FlashNextBf16FloatLibrary.statistics().contexts()!=0 || FlashNextBf16FloatLibrary.statistics().plans()!=0 || qsa.statistics().reservedBytes()!=0 || experts.statistics().residentBytes()!=0 || fusion.statistics().residentBytes()!=0 || layers.statistics().residentBytes()!=0
                    || FlashNextRmsNormLibrary.statistics().contexts()!=0 || FlashNextFp8Library.statistics().contexts()!=0 || FlashNextFp8Library.statistics().plans()!=0 || FlashNextFp8Library.registeredWeights()!=0) throw new AssertionError("MTP native resources leaked");
            var result=new LinkedHashMap<String,Object>();result.put("teacher",teacher);result.put("drafts",drafts);result.put("rollback_exact",true);result.put("reset_exact",true);result.put("failure_retry_exact",true);
            result.put("draft_steps",3);result.put("native_calls",FlashNextFp8Library.statistics().calls());result.put("moe_forwards",gpuExperts.forwards());
            result.put("qsa_forwards",gpuQsa.forwards());result.put("qsa_restores",gpuQsa.restores());result.put("kv_download_bytes",gpuQsa.kvHostDownloadBytes());
            result.put("fusion_forwards",gpuFusion.forwards());result.put("fusion_peak_bytes",gpuFusion.peakBytes());result.put("moe_peak_bytes",gpuExperts.peakBytes());result.put("resources_after_close",0);
            System.out.println(Json.write(result));
        }
    }
}

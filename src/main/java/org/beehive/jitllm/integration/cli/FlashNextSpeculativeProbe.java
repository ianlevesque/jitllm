package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.Qwen4CpuEngine;
import org.beehive.jitllm.backend.cpu.Qwen4SpeculativeSequence;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.server.Json;
import org.beehive.jitllm.backend.tornado.Qwen4GpuModel;
import org.beehive.jitllm.backend.tornado.FlashNextNvfp4Library;
import org.beehive.jitllm.backend.tornado.FlashNextBf16FloatLibrary;
import org.beehive.jitllm.backend.tornado.FlashNextFp8Library;
import org.beehive.jitllm.backend.tornado.FlashNextXqaLibrary;
import org.beehive.jitllm.backend.tornado.FlashNextSharedBf16Library;
import org.beehive.jitllm.backend.tornado.FlashNextRmsNormLibrary;

/** Real target-derived MTP generation against the same model's target-only decoding. */
public final class FlashNextSpeculativeProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextSpeculativeProbe MODEL_DIR REQUEST.json");
        System.out.println(Json.write(run(Path.of(args[0]),Json.parseObject(Files.readString(Path.of(args[1]))))));
    }
    private static List<Double> values(float[] input) {var result=new ArrayList<Double>();for(float v:input) result.add((double)v);return result;}
    private static List<Integer> ids(Object value) {return ((List<?>)value).stream().map(v->((Number)v).intValue()).toList();}
    private static final class Fixture implements Qwen4SpeculativeSequence.Observer {
        final List<Integer> tokens=new ArrayList<>(),drafts=new ArrayList<>();
        final List<List<Double>> hidden=new ArrayList<>();boolean frozen;
        public void teacher(int token,float[] targetHidden,Qwen4CpuEngine.Output output) {
            if(!frozen) {tokens.add(token);hidden.add(values(targetHidden));}
        }
        public void proposal(int token,float[] draftHidden,Qwen4CpuEngine.Output output) {
            frozen=true;if(drafts.size()<2) {drafts.add(token);if(drafts.size()==2) drafts.add(output.greedy());}
        }
    }
    public static Map<String,Object> run(Path checkpoint,Map<String,Object> request) throws Exception {
        System.setProperty("jitllm.qwen4.device",String.valueOf(((Number)request.getOrDefault("device",1)).intValue()));
        System.setProperty("jitllm.qwen4.mtp","true");
        int context=((Number)request.getOrDefault("context_length",4096)).intValue();
        int count=((Number)request.getOrDefault("max_tokens",8)).intValue();var prompt=ids(request.get("input_tokens"));
        float temperature=((Number)request.getOrDefault("temperature",0)).floatValue(),topP=((Number)request.getOrDefault("top_p",.9)).floatValue();
        long seed=((Number)request.getOrDefault("seed",42)).longValue();int bound=prompt.size()+count;
        var report=new LinkedHashMap<String,Object>();long start=System.nanoTime();
        var model=Qwen4GpuModel.load(checkpoint,context,1L<<20);
        try {
            try(var state=(Qwen4State)model.createNewState()) {
                var sequence=(Qwen4SpeculativeSequence)state.sequence();var fixture=new Fixture();sequence.observe(fixture);
                var streamed=new ArrayList<Integer>();
                var generated=model.generateOnDevice(state,0,prompt,Set.of(),bound,
                        Sampler.selectSampler(model.configuration().vocabularySize(),temperature,topP,seed),streamed::add);
                if(!generated.equals(streamed) || sequence.position()!=bound) throw new AssertionError("Speculative stream/position mismatch");
                if(request.containsKey("expected_tokens") && !generated.equals(ids(request.get("expected_tokens"))))
                    throw new AssertionError("Speculative output differs from staged reference: "+generated);
                var s=sequence.statistics();if(s.rounds()==0 || s.proposed()==0 || s.teacherForwards()==0 || s.draftForwards()==0)
                    throw new AssertionError("MTP proposal/verification did not execute");
                report.put("generated_tokens",generated);report.put("text",model.tokenizer().decode(generated));
                report.put("proposed",s.proposed());report.put("accepted",s.accepted());report.put("corrections",s.corrections());
                report.put("rounds",s.rounds());report.put("target_forwards",s.targetForwards());report.put("teacher_forwards",s.teacherForwards());report.put("draft_forwards",s.draftForwards());
                System.err.println("Speculative generated "+generated+"; "+s);
                if(request.containsKey("teacher_fixture_output")) {
                    if(fixture.drafts.size()!=3) throw new AssertionError("First draft-three fixture was not captured");
                    Files.writeString(Path.of((String)request.get("teacher_fixture_output")),Json.write(Map.of("tokens",fixture.tokens,"target_hidden",fixture.hidden,"draft_tokens",fixture.drafts)));
                }
                try(var baseline=new Qwen4State(model.configuration(),-1,model.engine().newSequence())) {
                    baseline.latestToken=-1;
                    var ordinary=model.generateOnDevice(baseline,0,prompt,Set.of(),bound,
                            Sampler.selectSampler(model.configuration().vocabularySize(),temperature,topP,seed),null);
                    if(!generated.equals(ordinary)) throw new AssertionError("MTP and target-only outputs differ: "+generated+" / "+ordinary);
                    report.put("target_only_tokens",ordinary);report.put("target_only_exact",true);
                }
                boolean resetReplay=!Boolean.FALSE.equals(request.get("reset_replay"));
                if(resetReplay) {
                    state.resetSequenceState();state.latestToken=-1;
                    var replay=model.generateOnDevice(state,0,prompt,Set.of(),bound,
                            Sampler.selectSampler(model.configuration().vocabularySize(),temperature,topP,seed),null);
                    if(!generated.equals(replay)) throw new AssertionError("MTP reset replay differs");
                }
                report.put("temperature",(double)temperature);report.put("top_p",(double)topP);report.put("seed",seed);
                report.put("reset_replay_checked",resetReplay);report.put("reset_replay_exact",resetReplay);
                report.put("execution",model.prepareDevice(state).toString());
                report.put("device_statistics",model.deviceStatistics());report.put("ple_reader_bytes",model.engine().statistics().residentBytes());
            }
        } finally {model.close();}
        var closed=model.deviceStatistics();
        for(String key:List.of("projection_bytes","recurrence_bytes","layer_bytes","qsa_reserved_bytes","moe_bytes","mtp_layer_bytes","mtp_qsa_reserved_bytes","mtp_expert_bytes","mtp_fusion_bytes"))
            if(((Number)closed.get(key)).longValue()!=0) throw new AssertionError("Resource survived close: "+key);
        if(FlashNextNvfp4Library.statistics().contexts()!=0 || FlashNextBf16FloatLibrary.statistics().contexts()!=0
                || FlashNextFp8Library.statistics().contexts()!=0 || FlashNextXqaLibrary.statistics().contexts()!=0
                || FlashNextSharedBf16Library.statistics().contexts()!=0 || FlashNextRmsNormLibrary.statistics().contexts()!=0
                || FlashNextNvfp4Library.registeredWeights()!=0 || FlashNextFp8Library.registeredWeights()!=0)
            throw new AssertionError("Speculative native resources survived close");
        report.put("native_fp8_calls",FlashNextFp8Library.statistics().calls());report.put("native_xqa_calls",FlashNextXqaLibrary.statistics().calls());
        report.put("resources_after_close",closed);report.put("elapsed_seconds",(System.nanoTime()-start)/1e9);report.put("passed",true);
        return report;
    }
}

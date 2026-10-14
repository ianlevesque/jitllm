package org.beehive.jitllm.backend.cpu;

import java.io.ByteArrayOutputStream;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.MetadataJson;
import org.beehive.jitllm.inference.CancellableTokenConsumer;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.MultimodalPromptPlan;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.vision.ImageGrid;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Caller wiring for batched prefill on a synthetic dim-16 checkpoint: the model path
 * with the property on must generate exactly what the sequential shared loop
 * generates (the host engine is bit-identical between the two), across chunk
 * boundaries, continuations, small prompts, stops, cancellation, a staged image
 * prompt, and mid-prompt failures.
 */
public class Qwen4PrefillModelWiringTest {
    private static final String CONFIG="""
        {"model_type":"qwen4_exp","text_config":{
          "hidden_size":16,"moe_intermediate_size":16,"num_hidden_layers":2,
          "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
          "vocab_size":32,"max_position_embeddings":128,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_type":"default","rope_theta":10000,"partial_rotary_factor":0.5},
          "hc_count":2,"hc_lowrank":2,"num_experts":4,"num_experts_per_tok":2,
          "shared_expert_intermediate_size":16,"norm_topk_prob":true,
          "linear_num_key_heads":2,"linear_num_value_heads":4,
          "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4,
          "ngram_size":3,"heads_per_ngram":2,"ple_embed_dim":16,"ple_conv_kernel_size":4,
          "ple_layer_ids":[2],"layer_types":["linear_attention","full_attention"],
          "indexer_budget":8,"indexer_compress_ratio":4,"indexer_head_dim":4,"indexer_n_heads":2,"indexer_kv_heads":1,
          "mtp_num_hidden_layers":1,"eos_token_id":1,"output_gate_type":"sigmoid","hidden_act":"silu"}}
        """;

    private static float pattern(int row,int column,int salt) {return (((row*37+column*11+salt*13)%17)-8)*0.0625f;}
    private static float[][] matrix(int rows,int columns,int salt) {
        float[][] values=new float[rows][columns];
        for(int i=0;i<rows;i++) for(int j=0;j<columns;j++) values[i][j]=pattern(i,j,salt);
        return values;
    }

    private static final class Checkpoint implements AutoCloseable {
        final Path directory=Files.createTempDirectory("qwen4-prefill-wiring-");
        final Map<String,Object> header=new LinkedHashMap<>();
        final ByteArrayOutputStream data=new ByteArrayOutputStream();
        Checkpoint() throws Exception {
            Files.writeString(directory.resolve("config.json"),CONFIG);
            var vocab=new StringBuilder();
            // The shared Qwen tokenizer base splits special tokens at <|endoftext|>.
            for(int i=0;i<32;i++) {if(i>0) vocab.append(',');vocab.append(i==31 ? "\"<|endoftext|>\"" : "\"t"+i+"\"").append(':').append(i);}
            Files.writeString(directory.resolve("tokenizer.json"),
                "{\"model\":{\"type\":\"BPE\",\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",\"vocab\":{"
                +vocab+"},\"merges\":[]},\"added_tokens\":[],\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                +"{\"type\":\"Split\",\"behavior\":\"Isolated\",\"pattern\":{\"Regex\":\".\"}},{\"type\":\"ByteLevel\"}]}}");
            String root="model.language_model";
            bf(root+".embed_tokens.weight",matrix(32,16,41));
            bf("lm_head.weight",matrix(32,16,42));
            hyper(root+".hyper_connection_mixer",false,43);
            for(int layer=0;layer<2;layer++) {
                String prefix=root+".layers."+layer;
                hyper(prefix+".attn_hyper_connection",true,50+layer*10);
                hyper(prefix+".mlp_hyper_connection",true,51+layer*10);
                moe(prefix+".mlp",52+layer*10);
            }
            String gdn=root+".layers.0.linear_attn";
            bf(gdn+".in_proj_qkv.weight",matrix(32,16,7));bf(gdn+".in_proj_z.weight",matrix(16,16,8));
            bf(gdn+".in_proj_a.weight",matrix(4,16,9));bf(gdn+".in_proj_b.weight",matrix(4,16,10));
            bf(gdn+".out_proj.weight",matrix(16,16,11));bf(gdn+".conv1d.weight",matrix(32,4,12));
            bf(gdn+".norm.weight",new float[][]{{1,1,1,1}});
            bf(gdn+".A_log",matrix(1,4,13));bf(gdn+".dt_bias",matrix(1,4,14));
            String qsa=root+".layers.1.self_attn";
            bf(qsa+".q_proj.weight",matrix(32,16,15));bf(qsa+".k_proj.weight",matrix(8,16,16));bf(qsa+".v_proj.weight",matrix(8,16,17));
            bf(qsa+".o_proj.weight",matrix(16,16,18));bf(qsa+".indexer.index_qk_proj.weight",matrix(12,16,19));
            bf(qsa+".q_norm.weight",new float[1][8]);bf(qsa+".k_norm.weight",new float[1][8]);
            bf(qsa+".indexer.q_layernorm.weight",new float[1][4]);bf(qsa+".indexer.k_layernorm.weight",new float[1][4]);
            String ple=root+".layers.1.ple.ple_embedding";
            longs(ple+".layer_multipliers",3,5,7);longs(ple+".ngram_heads_vocab_sizes",3,3,3,3);longs(ple+".ngram_heads_offsets",0,3,6,9);
            bf(ple+".ngram_embedding.weight_scale",new float[][]{{1}});
            byte[] pleRows=new byte[48];Arrays.fill(pleRows,(byte)0x30);add(ple+".ngram_embedding.weight","F8_E4M3",List.of(12,4),pleRows);
            String pleProjections=root+".layers.1.ple";
            bf(pleProjections+".key_proj.weight",matrix(32,16,4));bf(pleProjections+".value_proj.weight",matrix(16,16,5));
            for(String name:List.of("norm_key","norm_query","norm_conv")) bf(pleProjections+"."+name+".weight",new float[1][32]);
            bf(pleProjections+".conv1d.weight",matrix(32,4,6));
            byte[] metadata=MetadataJson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            var file=ByteBuffer.allocate(8+metadata.length+data.size()).order(ByteOrder.LITTLE_ENDIAN);
            file.putLong(metadata.length).put(metadata).put(data.toByteArray());Files.write(directory.resolve("model.safetensors"),file.array());
        }
        void hyper(String prefix,boolean inject,int salt) throws Exception {
            bf(prefix+".hc_norm.weight",new float[1][32]);
            bf(prefix+".input_mix_weight_down.weight",matrix(2,32,salt));
            bf(prefix+".input_mix_weight_up.weight",matrix(32,2,salt+1));
            if(inject) bf(prefix+".block_inject_weight.weight",matrix(2,32,salt+2));
        }
        void moe(String prefix,int salt) throws Exception {
            bf(prefix+".gate.weight",matrix(4,16,salt));
            bf(prefix+".shared_expert.gate_proj.weight",matrix(16,16,salt+1));
            bf(prefix+".shared_expert.up_proj.weight",matrix(16,16,salt+2));
            bf(prefix+".shared_expert.down_proj.weight",matrix(16,16,salt+3));
            bf(prefix+".shared_expert_gate.weight",matrix(1,16,salt+4));
            for(int expert=0;expert<4;expert++)
                for(String kind:List.of("gate_proj","up_proj","down_proj")) nvfp4(prefix+".experts."+expert+"."+kind,16,16,expert*3+kind.length());
        }
        void add(String name,String dtype,List<Integer> shape,byte[] bytes) throws Exception {
            int start=data.size();data.write(bytes);header.put(name,Map.of("dtype",dtype,"shape",shape,"data_offsets",List.of(start,data.size())));
        }
        void bf(String name,float[][] values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*values[0].length*2).order(ByteOrder.LITTLE_ENDIAN);
            for(var row:values) for(float value:row) bytes.putShort((short)(Float.floatToRawIntBits(value)>>>16));
            add(name,"BF16",List.of(values.length,values[0].length),bytes.array());
        }
        void longs(String name,long... values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*8).order(ByteOrder.LITTLE_ENDIAN);for(long value:values) bytes.putLong(value);
            add(name,"I64",List.of(values.length),bytes.array());
        }
        void f32(String name,float value) throws Exception {
            add(name,"F32",List.of(1),ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array());
        }
        void nvfp4(String prefix,int rows,int columns,int salt) throws Exception {
            byte[] packed=new byte[rows*columns/2];
            for(int i=0;i<packed.length;i++) packed[i]=(byte)(i*7+salt*31+3);
            add(prefix+".weight","U8",List.of(rows,columns/2),packed);
            byte[] scales=new byte[rows*(columns/16)];Arrays.fill(scales,(byte)0x38);
            add(prefix+".weight_scale","F8_E4M3",List.of(rows,columns/16),scales);
            f32(prefix+".weight_scale_2",1);f32(prefix+".input_scale",1);
        }
        public void close() throws Exception {
            try(var files=Files.walk(directory)) {for(var file:files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);}
        }
    }

    /** The engine captures the properties at construction, so set them per engine. */
    private static Qwen4CpuModel model(Path directory,boolean batched,int rows) throws Exception {
        if(batched) {System.setProperty("jitllm.qwen4.batchedPrefill","true");System.setProperty("jitllm.qwen4.prefillRows",Integer.toString(rows));}
        else {System.clearProperty("jitllm.qwen4.batchedPrefill");System.clearProperty("jitllm.qwen4.prefillRows");}
        try {return new Qwen4CpuModel(new Qwen4CpuEngine(directory,-1,1<<20,null));}
        finally {System.clearProperty("jitllm.qwen4.batchedPrefill");System.clearProperty("jitllm.qwen4.prefillRows");}
    }
    private static List<Integer> prompt(int count) {
        var tokens=new ArrayList<Integer>();for(int i=0;i<count;i++) tokens.add((i*7+3)%30+2);return tokens;
    }
    private static Qwen4CpuEngine.Sequence sequence(State state) {
        return (Qwen4CpuEngine.Sequence)((Qwen4State)state).sequence();
    }
    private record Run(Qwen4CpuModel model,State state,List<Integer> generated,List<Integer> callbacks) {}
    private static Run generate(Qwen4CpuModel model,List<Integer> prompt,int budget,Set<Integer> stops) {
        var state=model.createNewState();var callbacks=new ArrayList<Integer>();
        var generated=model.generateTokens(state,0,prompt,stops,budget,Sampler.TENSOR_ARGMAX,false,callbacks::add);
        return new Run(model,state,generated,callbacks);
    }

    @Test public void batchedMatchesSequentialAcrossChunksAndContinuation() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var onSmall=model(checkpoint.directory,true,4);
            var onWide=model(checkpoint.directory,true,64)) {
            var runs=new ArrayList<Run>();
            for(var model:List.of(off,onSmall,onWide)) {
                var run=generate(model,prompt(40),48,Set.of());
                runs.add(run);
                assertEquals(8,run.generated.size());assertEquals(run.generated,run.callbacks);
                assertEquals(47,sequence(run.state).position());
                assertEquals((int)run.generated.getLast(),run.state.latestToken);
                // Second turn: the API position counts the pending last token, which the
                // model must ingest (sequentially) before the new prompt's chunks.
                var more=new ArrayList<Integer>();
                var second=model.generateTokens(run.state,48,prompt(20),Set.of(),74,Sampler.TENSOR_ARGMAX,false,more::add);
                assertEquals(6,second.size());assertEquals(second,more);assertEquals(73,sequence(run.state).position());
                run.generated.addAll(second);
            }
            assertEquals(runs.get(0).generated,runs.get(1).generated);
            assertEquals(runs.get(0).generated,runs.get(2).generated);
            assertEquals(off.engine().forwardStatistics(),onSmall.engine().forwardStatistics());
            assertEquals(0,off.engine().batchedPrefillStatistics().chunks());
            assertTrue(onSmall.engine().batchedPrefillStatistics().chunks()>1);
            assertTrue(onSmall.engine().batchedPrefillStatistics().rows()>=39);
            assertTrue(onWide.engine().batchedPrefillStatistics().chunks()>=1);
            assertTrue(onSmall.engine().batchedPrefillStatistics().chunks()>onWide.engine().batchedPrefillStatistics().chunks());
            // The split loop counted a one-token prompt; the model restores the real
            // count of the last call (the 20-token continuation) in the metrics.
            assertEquals(20,org.beehive.jitllm.auxiliary.RunMetrics.snapshot().promptEvalCount());
        }
    }

    @Test public void smallPromptsStaySequentialAndMatch() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            for(int size:new int[]{1,2,10,16,17}) {
                long chunksBefore=on.engine().batchedPrefillStatistics().chunks();
                var expected=generate(off,prompt(size),size+5,Set.of());
                var actual=generate(on,prompt(size),size+5,Set.of());
                assertEquals("prompt size "+size,expected.generated,actual.generated);
                assertEquals(sequence(expected.state).position(),sequence(actual.state).position());
                long chunks=on.engine().batchedPrefillStatistics().chunks()-chunksBefore;
                if(size<=16) assertEquals("fused geometry must not batch at "+size,0,chunks);
                else assertTrue("17-token prompt batches",chunks>0);
            }
        }
    }

    @Test public void stopTokensAndCancellationMatch() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            int first=generate(off,prompt(24),40,Set.of()).generated.getFirst();
            for(var model:List.of(off,on)) {
                var stopped=generate(model,prompt(24),40,Set.of(first));
                assertEquals(List.of(first),stopped.generated);
                assertEquals(24,sequence(stopped.state).position());
                var state=model.createNewState();var seen=new ArrayList<Integer>();
                var consumer=new CancellableTokenConsumer() {
                    public void accept(int token) {seen.add(token);}
                    public boolean cancellationRequested() {return seen.size()==2;}
                };
                var cancelled=model.generateTokens(state,0,prompt(24),Set.of(),40,Sampler.TENSOR_ARGMAX,false,consumer);
                assertEquals(2,cancelled.size());assertEquals(seen,cancelled);
                assertEquals((int)cancelled.getLast(),state.latestToken);
            }
        }
    }

    @Test public void echoOutputMatches() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            var original=System.err;String expected=null;
            for(var model:List.of(off,on)) {
                var captured=new ByteArrayOutputStream();
                try {
                    System.setErr(new java.io.PrintStream(captured));
                    var state=model.createNewState();
                    model.generateTokens(state,0,prompt(24),Set.of(),28,Sampler.TENSOR_ARGMAX,true,null);
                } finally {System.setErr(original);}
                String text=captured.toString();
                assertFalse(text.isEmpty());
                if(expected==null) expected=text;else assertEquals(expected,text);
            }
        }
    }

    @Test public void stagedImagePromptDelegatesAndMatches() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            // Token 1 never occurs in the synthetic prompt pattern, so it can stand
            // in for the image placeholder id.
            var tokens=prompt(24);tokens.set(6,1);tokens.set(7,1);
            int[] ids=tokens.stream().mapToInt(Integer::intValue).toArray();
            float[][] rows=new float[2][16];
            for(int r=0;r<2;r++) for(int c=0;c<16;c++) rows[r][c]=pattern(r,c,99);
            var plan=MultimodalPromptPlan.of(ids,List.of(new MultimodalPromptPlan.ImageSpan(6,new ImageGrid(1,4,2),rows)),1,2);
            List<Integer> expected=null;
            for(var model:List.of(off,on)) {
                var state=model.createNewState();
                ((Qwen4State)state).stageMultimodalPrompt(plan);
                var generated=model.generateTokens(state,0,tokens,Set.of(),29,Sampler.TENSOR_ARGMAX,false,null);
                if(expected==null) expected=generated;else assertEquals(expected,generated);
                assertEquals(28,sequence(state).position());
            }
            // A staged plan forces sequential spliced forwards even with the property on.
            assertEquals(0,on.engine().batchedPrefillStatistics().chunks());
            assertTrue(on.engine().batchedPrefillStatistics().fallbacks()>0);
        }
    }

    @Test public void invalidTokenMidPromptRollsBackToChunkStartAndRetryMatches() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            var bad=prompt(40);bad.set(10,99);
            var retries=new ArrayList<List<Integer>>();
            for(var model:List.of(off,on)) {
                var state=model.createNewState();var seen=new ArrayList<Integer>();
                assertThrows(IllegalArgumentException.class,
                    ()->model.generateTokens(state,0,bad,Set.of(),48,Sampler.TENSOR_ARGMAX,false,seen::add));
                assertTrue(seen.isEmpty());
                // finishPrefill ran in the finally: no prefill window remains open.
                assertTrue(sequence(state).requiresLogits());
                int position=sequence(state).position();
                // Sequential keeps the 10 completed rows; the chunked path validates
                // the whole 4-row slice up front and keeps only completed slices.
                assertEquals(model==off ? 10 : 8,position);
                var rest=prompt(40).subList(position,40);
                retries.add(model.generateTokens(state,position,rest,Set.of(),48,Sampler.TENSOR_ARGMAX,false,null));
                assertEquals(47,sequence(state).position());
            }
            assertEquals(retries.get(0),retries.get(1));
            // An invalid final token fails in the shared loop on both paths, at 39.
            var badLast=prompt(40);badLast.set(39,99);
            for(var model:List.of(off,on)) {
                var state=model.createNewState();
                assertThrows(IllegalArgumentException.class,
                    ()->model.generateTokens(state,0,badLast,Set.of(),48,Sampler.TENSOR_ARGMAX,false,null));
                assertEquals(39,sequence(state).position());
                assertTrue(sequence(state).requiresLogits());
            }
        }
    }

    @Test public void truncatedPromptAndPositionMismatchKeepLoopSemantics() throws Exception {
        try(var checkpoint=new Checkpoint();
            var off=model(checkpoint.directory,false,32);
            var on=model(checkpoint.directory,true,4)) {
            for(var model:List.of(off,on)) {
                // maxTokens ends inside the prompt: the loop ingests only up to it.
                var state=model.createNewState();
                var generated=model.generateTokens(state,0,prompt(40),Set.of(),25,Sampler.TENSOR_ARGMAX,false,null);
                assertTrue(generated.isEmpty());assertEquals(25,sequence(state).position());
                assertTrue(sequence(state).requiresLogits());
                // Position mismatch still throws before any ingest, prefill finished once.
                var mismatch=model.createNewState();
                assertThrows(IllegalArgumentException.class,
                    ()->model.generateTokens(mismatch,3,prompt(4),Set.of(),10,Sampler.TENSOR_ARGMAX,false,null));
                assertEquals(0,sequence(mismatch).position());
            }
            assertEquals(0,on.engine().batchedPrefillStatistics().chunks());
        }
    }
}

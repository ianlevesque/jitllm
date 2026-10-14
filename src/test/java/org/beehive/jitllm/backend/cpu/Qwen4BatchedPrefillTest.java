package org.beehive.jitllm.backend.cpu;

import java.io.ByteArrayOutputStream;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Batched-prefill equivalence on a synthetic checkpoint: on the host reference every
 * batched forward must reproduce the sequential forwards bit for bit, including the
 * state a continuation forward observes. (On a GEMM executor only the GEMM geometry
 * changes; low-bit rounding there is qualified by the GPU greedy-parity probes.)
 */
public class Qwen4BatchedPrefillTest {
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

    private static float pattern(int row,int column,int salt) {
        return (((row*37+column*11+salt*13)%17)-8)*0.0625f;
    }
    private static float[][] matrix(int rows,int columns,int salt) {
        float[][] values=new float[rows][columns];
        for(int i=0;i<rows;i++) for(int j=0;j<columns;j++) values[i][j]=pattern(i,j,salt);
        return values;
    }
    private static float[][] rows(int count,int width,int salt) {
        float[][] values=new float[count][];
        for(int i=0;i<count;i++) {values[i]=new float[width];for(int j=0;j<width;j++) values[i][j]=pattern(i,j,salt)+1;}
        return values;
    }

    private static final class Fixture implements AutoCloseable {
        final Path directory=Files.createTempDirectory("qwen4-batched-prefill-");
        final Map<String,Object> header=new LinkedHashMap<>();
        final ByteArrayOutputStream data=new ByteArrayOutputStream();
        final PagedTensorReader reader;
        final Qwen4Configuration config;
        final SafeTensorIndex index;
        final Qwen4CpuLayers.HyperConnection hyper;
        final Qwen4CpuLayers.Ple ple;
        final Qwen4DeltaNet delta;
        final Qwen4QsaAttention qsa;
        final Qwen4Moe moe;
        Fixture() throws Exception {
            Files.writeString(directory.resolve("config.json"),CONFIG);
            bf("test.hyper.hc_norm.weight",new float[1][32]);
            bf("test.hyper.input_mix_weight_down.weight",matrix(2,32,1));
            bf("test.hyper.input_mix_weight_up.weight",matrix(32,2,2));
            bf("test.hyper.block_inject_weight.weight",matrix(2,32,3));
            String table="test.ple.ple_embedding";
            longs(table+".layer_multipliers",3,5,7);longs(table+".ngram_heads_vocab_sizes",3,3,3,3);longs(table+".ngram_heads_offsets",0,3,6,9);
            bf(table+".ngram_embedding.weight_scale",new float[][]{{1}});
            byte[] pleRows=new byte[48];Arrays.fill(pleRows,(byte)0x30);add(table+".ngram_embedding.weight","F8_E4M3",List.of(12,4),pleRows);
            bf("test.ple.key_proj.weight",matrix(32,16,4));bf("test.ple.value_proj.weight",matrix(16,16,5));
            for(String name:List.of("norm_key","norm_query","norm_conv")) bf("test.ple."+name+".weight",new float[1][32]);
            bf("test.ple.conv1d.weight",matrix(32,4,6));
            bf("test.gdn.in_proj_qkv.weight",matrix(32,16,7));bf("test.gdn.in_proj_z.weight",matrix(16,16,8));
            bf("test.gdn.in_proj_a.weight",matrix(4,16,9));bf("test.gdn.in_proj_b.weight",matrix(4,16,10));
            bf("test.gdn.out_proj.weight",matrix(16,16,11));bf("test.gdn.conv1d.weight",matrix(32,4,12));
            bf("test.gdn.norm.weight",new float[][]{{1,1,1,1}});
            bf("test.gdn.A_log",matrix(1,4,13));bf("test.gdn.dt_bias",matrix(1,4,14));
            bf("test.q_proj.weight",matrix(32,16,15));bf("test.k_proj.weight",matrix(8,16,16));bf("test.v_proj.weight",matrix(8,16,17));
            bf("test.o_proj.weight",matrix(16,16,18));bf("test.indexer.index_qk_proj.weight",matrix(12,16,19));
            bf("test.q_norm.weight",new float[1][8]);bf("test.k_norm.weight",new float[1][8]);
            bf("test.indexer.q_layernorm.weight",new float[1][4]);bf("test.indexer.k_layernorm.weight",new float[1][4]);
            bf("test.mlp.gate.weight",matrix(4,16,20));
            bf("test.mlp.shared_expert.gate_proj.weight",matrix(16,16,21));
            bf("test.mlp.shared_expert.up_proj.weight",matrix(16,16,22));
            bf("test.mlp.shared_expert.down_proj.weight",matrix(16,16,23));
            bf("test.mlp.shared_expert_gate.weight",matrix(1,16,24));
            for(int expert=0;expert<4;expert++)
                for(String kind:List.of("gate_proj","up_proj","down_proj")) nvfp4("test.mlp.experts."+expert+"."+kind,16,16,expert*3+kind.length());
            byte[] metadata=MetadataJson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            var file=ByteBuffer.allocate(8+metadata.length+data.size()).order(ByteOrder.LITTLE_ENDIAN);
            file.putLong(metadata.length).put(metadata).put(data.toByteArray());Files.write(directory.resolve("model.safetensors"),file.array());
            reader=new PagedTensorReader(1<<20,64);index=SafeTensorIndex.open(directory);config=Qwen4Configuration.load(directory,-1);
            hyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,"test.hyper",true);
            ple=new Qwen4CpuLayers.Ple(config,index,reader,"test.ple");
            delta=new Qwen4DeltaNet(config,index,reader,"test.gdn");
            qsa=new Qwen4QsaAttention(config,index,reader,"test",true,true);
            moe=new Qwen4Moe(config,index,reader,"test.mlp");
        }
        void add(String name,String dtype,List<Integer> shape,byte[] bytes) throws Exception {
            int start=data.size();data.write(bytes);header.put(name,Map.of("dtype",dtype,"shape",shape,"data_offsets",List.of(start,data.size())));
        }
        void bf(String name,float[][] values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*values[0].length*2).order(ByteOrder.LITTLE_ENDIAN);
            for(var row:values) for(float value:row) bytes.putShort((short)(Float.floatToRawIntBits(value)>>>16));
            add(name,"BF16",List.of(values.length,values[0].length),bytes.array());
        }
        void f32(String name,float value) throws Exception {
            add(name,"F32",List.of(1),ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array());
        }
        void longs(String name,long... values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*8).order(ByteOrder.LITTLE_ENDIAN);for(long value:values) bytes.putLong(value);
            add(name,"I64",List.of(values.length),bytes.array());
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
            hyper.close();ple.close();moe.close();reader.close();
            try(var files=Files.walk(directory)) {for(var file:files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);}
        }
    }

    @Test public void chunkPlannerRespectsGeometryFreshAndPrefillBoundaries() {
        var plan=Qwen4CpuEngine.BatchedPrefill.class;
        // Legacy geometry (chunk property 1), 32-row cap, fresh through 4096.
        assertEquals(32,Qwen4CpuEngine.BatchedPrefill.chunkLength(0,0,100,100,32,1));
        assertEquals(32,Qwen4CpuEngine.BatchedPrefill.chunkLength(64,0,100,100,32,1));
        assertEquals(4,Qwen4CpuEngine.BatchedPrefill.chunkLength(96,0,100,100,32,1));
        assertEquals(0,Qwen4CpuEngine.BatchedPrefill.chunkLength(99,0,100,100,32,1));
        assertEquals(0,Qwen4CpuEngine.BatchedPrefill.chunkLength(100,0,100,100,32,1));
        // Fresh boundary splits the chunk; the row after it starts a non-fresh chunk.
        assertEquals(10,Qwen4CpuEngine.BatchedPrefill.chunkLength(90,0,200,100,32,1));
        assertEquals(32,Qwen4CpuEngine.BatchedPrefill.chunkLength(100,0,200,100,32,1));
        // Fused geometry (<=16 reported rows) never batches.
        assertEquals(0,Qwen4CpuEngine.BatchedPrefill.chunkLength(0,0,16,16,32,1));
        assertEquals(0,Qwen4CpuEngine.BatchedPrefill.chunkLength(0,0,10,10,32,1));
        assertEquals(17,Qwen4CpuEngine.BatchedPrefill.chunkLength(0,0,17,17,32,1));
        // The 4096 geometry boundary splits even an uncapped chunk.
        assertEquals(96,Qwen4CpuEngine.BatchedPrefill.chunkLength(4000,0,9000,4096,4096,1));
        assertEquals(32,Qwen4CpuEngine.BatchedPrefill.chunkLength(4096,0,9000,4096,32,1));
        // A small geometry chunk property keeps its own alignment.
        assertEquals(32,Qwen4CpuEngine.BatchedPrefill.chunkLength(10,10,200,0,64,32));
        assertEquals(2,Qwen4CpuEngine.BatchedPrefill.chunkLength(40,10,200,0,64,32));
        assertEquals(4096,Qwen4CpuEngine.BatchedPrefill.effectiveRows(100000));
        assertThrows(IllegalArgumentException.class,()->Qwen4CpuEngine.BatchedPrefill.effectiveRows(0));
        assertNotNull(plan);
    }

    @Test public void hyperMixAndCombineBatchMatchSequentialRows() throws Exception {
        try(var fixture=new Fixture()) {
            float[][] inputs=rows(20,32,31);
            var batched=fixture.hyper.mixBatch(inputs,32);
            float[][] blocks=rows(20,16,32);
            var combined=fixture.hyper.combineBatch(blocks,batched);
            for(int row=0;row<inputs.length;row++) {
                var expected=fixture.hyper.mix(inputs[row],32);
                assertArrayEquals(expected.output(),batched[row].output(),0);
                assertArrayEquals(expected.residual(),batched[row].residual(),0);
                assertArrayEquals(expected.normalized(),batched[row].normalized(),0);
                assertArrayEquals(fixture.hyper.combine(blocks[row],expected),combined[row],0);
            }
            assertThrows(IllegalArgumentException.class,()->fixture.hyper.mixBatch(inputs,16));
        }
    }

    @Test public void pleBatchMatchesSequentialHistoryAndConvolution() throws Exception {
        try(var fixture=new Fixture()) {
            int[] tokens={2,3,2,5,3,2,7,2,3,5,2,3};
            float[][] hiddens=rows(tokens.length,32,33);
            var sequential=fixture.ple.newState();var batched=fixture.ple.newState();
            float[][] outputs=fixture.ple.applyBatch(tokens,hiddens,batched);
            for(int row=0;row<tokens.length;row++)
                assertArrayEquals(fixture.ple.apply(tokens[row],hiddens[row],sequential,false),outputs[row],0);
            // Continuation observes identical history/convolution state.
            float[] next=rows(1,32,34)[0];
            assertArrayEquals(fixture.ple.apply(3,next,sequential,false),fixture.ple.apply(3,next,batched,false),0);
        }
    }

    @Test public void deltaBatchMatchesSequentialRecurrenceAndSnapshot() throws Exception {
        try(var fixture=new Fixture()) {
            float[][] inputs=rows(20,16,35);
            var sequential=fixture.delta.newState();var batched=fixture.delta.newState();
            float[][] outputs=fixture.delta.forwardBatch(inputs,batched);
            for(int row=0;row<inputs.length;row++)
                assertArrayEquals(fixture.delta.forward(inputs[row],sequential),outputs[row],0);
            float[] next=rows(1,16,36)[0];
            assertArrayEquals(fixture.delta.forward(next,sequential),fixture.delta.forward(next,batched),0);
            // A snapshot taken after batched rows restores to the same continuation.
            var replay=fixture.delta.newState();
            fixture.delta.forwardBatch(Arrays.copyOf(inputs,10),replay);
            var saved=replay.snapshot();
            var expected=fixture.delta.forwardBatch(Arrays.copyOfRange(inputs,10,20),replay);
            replay.restore(saved);
            var actual=fixture.delta.forwardBatch(Arrays.copyOfRange(inputs,10,20),replay);
            for(int row=0;row<expected.length;row++) assertArrayEquals(expected[row],actual[row],0);
            saved.close();
        }
    }

    @Test public void qsaBatchMatchesSequentialAppendSelectionAndFreshRelease() throws Exception {
        try(var fixture=new Fixture()) {
            float[][] inputs=rows(20,16,37);
            var sequential=fixture.qsa.newState();var batched=fixture.qsa.newState();
            int[][] ropes=new int[inputs.length][];
            for(int row=0;row<ropes.length;row++) ropes[row]=new int[]{row,row,row};
            float[][] outputs=fixture.qsa.forwardBatch(inputs,ropes,batched,true,true);
            for(int row=0;row<inputs.length;row++)
                assertArrayEquals(fixture.qsa.forward(inputs[row],sequential,true,true),outputs[row],0);
            assertEquals(sequential.tokens(),batched.tokens());
            assertArrayEquals(sequential.selectedTokens(),batched.selectedTokens());
            sequential.finishFreshPrefill();batched.finishFreshPrefill();
            float[] next=rows(1,16,38)[0];
            assertArrayEquals(fixture.qsa.forward(next,sequential),fixture.qsa.forward(next,batched),0);
            assertArrayEquals(sequential.selectedTokens(),batched.selectedTokens());
            // Snapshot/restore across a batched chunk replays exactly.
            var saved=batched.snapshot();
            var expected=fixture.qsa.forward(next,batched);
            batched.restore(saved);
            assertArrayEquals(expected,fixture.qsa.forward(next,batched),0);
            saved.close();
        }
    }

    @Test public void moeBatchMatchesSequentialRoutingAndOutputs() throws Exception {
        try(var fixture=new Fixture()) {
            float[][] inputs=rows(8,16,39);
            var outputs=fixture.moe.forwardBatch(inputs);
            for(int row=0;row<inputs.length;row++) {
                var expected=fixture.moe.forward(inputs[row]);
                assertArrayEquals(expected.output(),outputs[row].output(),0);
                assertArrayEquals(expected.routing().experts(),outputs[row].routing().experts());
                assertArrayEquals(expected.routing().weights(),outputs[row].routing().weights(),0);
            }
        }
    }

    @Test public void projectBatchMatchesProjectRowForRow() throws Exception {
        try(var fixture=new Fixture()) {
            var matrix=Qwen4CpuLayers.matrix(fixture.index,fixture.reader,"test.gdn.in_proj_qkv",32,16);
            float[][] inputs=rows(9,16,40);
            float[] batched=Qwen4CpuLayers.projectBatch(matrix,Qwen4CpuLayers.flatten(inputs,16),inputs.length);
            float[][] split=Qwen4CpuLayers.unflatten(batched,inputs.length,32);
            for(int row=0;row<inputs.length;row++)
                assertArrayEquals(Qwen4CpuLayers.project(matrix,inputs[row]),split[row],0);
        }
    }
}

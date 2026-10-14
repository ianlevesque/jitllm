package org.beehive.jitllm.model.qwen4;

import static org.junit.Assert.*;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

public final class Qwen4ConfigurationTest {
    @Test public void hostStateAllocatesNoDenseContextCacheAndResetsItsOwnSequence() throws Exception {
        var config = load(CONFIG, -1);
        class Sequence implements org.beehive.jitllm.inference.state.Qwen4SequenceState {
            int position = 3;
            boolean closed;
            public int position() { return position; }
            public void reset() { position = 0; }
            public void close() { closed = true; }
        }
        var first = new Sequence(); var second = new Sequence();
        var state = org.beehive.jitllm.inference.state.State.withPrefillBatchSize(1,
                () -> new org.beehive.jitllm.inference.state.Qwen4State(config, -1, first));
        assertEquals(0, state.keyCache.length); assertEquals(0, state.valueCache.length);
        assertEquals(config.vocabularySize(), state.logits.size());
        assertThrows(CloneNotSupportedException.class, state::clone);
        state.resetSequenceState(); assertEquals(0, first.position()); assertEquals(3, second.position());
        state.close(); assertTrue(first.closed); assertFalse(second.closed);
    }
    static final String CONFIG = """
        {"model_type":"qwen4_exp","text_config":{
          "hidden_size":8,"moe_intermediate_size":4,"num_hidden_layers":2,
          "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
          "vocab_size":16,"max_position_embeddings":16,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_type":"default","rope_theta":10000,"partial_rotary_factor":0.5},
          "hc_count":4,"hc_lowrank":2,"num_experts":8,"num_experts_per_tok":2,
          "shared_expert_intermediate_size":4,"norm_topk_prob":true,
          "linear_num_key_heads":2,"linear_num_value_heads":4,
          "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4,
          "ngram_size":3,"heads_per_ngram":2,"ple_embed_dim":16,"ple_conv_kernel_size":4,
          "ple_layer_ids":[2],"layer_types":["linear_attention","full_attention"],
          "indexer_budget":8,"indexer_compress_ratio":4,"indexer_head_dim":4,"indexer_n_heads":2,"indexer_kv_heads":1,
          "mtp_num_hidden_layers":1,"eos_token_id":1,"output_gate_type":"sigmoid","hidden_act":"silu"}}
        """;

    private static Qwen4Configuration load(String text,int context) throws Exception {
        Path directory=Files.createTempDirectory("qwen4-config-test-");
        Path file=directory.resolve("config.json");
        try {
            Files.writeString(file,text);
            return Qwen4Configuration.load(directory,context);
        } finally { Files.deleteIfExists(file); Files.deleteIfExists(directory); }
    }

    @Test public void nativeContextTrunkAndMtpStaySeparate() throws Exception {
        var config=load(CONFIG,-1);
        assertEquals(16,config.contextLength()); assertEquals(2,config.numberOfLayers());
        assertEquals(1,config.mtpLayers()); assertEquals(1,config.keyValueLayerCount());
        assertTrue(config.isRecurrentLayer(0)); assertFalse(config.isRecurrentLayer(1));
        assertFalse(config.hasPle(0)); assertTrue(config.hasPle(1));
        assertEquals(1792L,config.recurrentStateBytes());
        var mtp=config.mtpConfiguration();
        assertEquals(1,mtp.numberOfLayers());assertEquals(1,mtp.keyValueLayerCount());
        assertEquals(0,mtp.mtpLayers());assertEquals(0,mtp.recurrentStateBytes());
        assertFalse(mtp.isRecurrentLayer(0));assertFalse(mtp.hasPle(0));
        assertEquals(config.hyperDim(),mtp.hyperDim());assertEquals(config.contextLength(),mtp.contextLength());
        assertEquals(2,config.numberOfLayers());assertTrue(config.hasPle(1));
    }
    @Test public void requestedContextKeepsNativeCeiling() throws Exception {
        var config=load(CONFIG,8);
        assertEquals(8,config.contextLength()); assertEquals(16,config.contextLengthModel());
        var refusal=assertThrows(java.io.IOException.class,()->load(CONFIG,17));
        assertTrue(refusal.getMessage(),refusal.getMessage().contains("17") && refusal.getMessage().contains("16"));
    }
    @Test public void zeroAndNegativeRequestsSelectTheNativeContextInFull() throws Exception {
        // 0 is ModelOptions' "the model's own"; a negative value is the loader's. Neither may
        // reach the record as a literal length.
        assertEquals(16,load(CONFIG,0).contextLength());
        assertEquals(16,load(CONFIG,-1).contextLength());
        assertEquals(16,load(CONFIG,16).contextLength());
    }
    @Test public void attentionDimensionsAreExplicit() throws Exception {
        var config=load(CONFIG,-1);
        assertEquals(8,config.headSize()); assertEquals(4,config.rotaryDimensions());
        assertEquals(8,config.kvDim()); assertEquals(32,config.hyperDim());
        assertEquals(32,config.linearQkvDim());
    }
    @Test public void invalidGroupingAndRoutingFailBeforeLoadingWeights() {
        assertThrows(IllegalArgumentException.class,()->load(CONFIG.replace("\"linear_num_value_heads\":4","\"linear_num_value_heads\":5"),-1));
        assertThrows(IllegalArgumentException.class,()->load(CONFIG.replace("\"num_experts_per_tok\":2","\"num_experts_per_tok\":9"),-1));
    }
    @Test public void pleLayerNumbersAreOneBasedAndUnique() {
        assertThrows(IllegalArgumentException.class,()->load(CONFIG.replace("\"ple_layer_ids\":[2]","\"ple_layer_ids\":[0]"),-1));
        assertThrows(IllegalArgumentException.class,()->load(CONFIG.replace("\"ple_layer_ids\":[2]","\"ple_layer_ids\":[2,2]"),-1));
    }
    @Test public void unsupportedRopeAndLayerKindAreRejected() {
        assertThrows(java.io.IOException.class,()->load(CONFIG.replace("\"rope_type\":\"default\"","\"rope_type\":\"yarn\""),-1));
        assertThrows(IllegalArgumentException.class,()->load(CONFIG.replace("\"linear_attention\"","\"other_attention\""),-1));
    }
}

package org.beehive.jitllm.model.qwen4;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.beehive.jitllm.format.safetensors.MetadataJson;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Dimensions of the Qwen4-Exp Flash-Next trunk, kept separate from its MTP block. */
public record Qwen4Configuration(int dim, int hiddenDim, int numberOfLayers,
        int numberOfHeads, int numberOfKeyValueHeads, int headSize,
        int vocabularySize, int contextLengthModel, int contextLength,
        float rmsNormEps, float ropeTheta, int rotaryDimensions,
        int hcCount, int hcLowrank, int experts, int expertsPerToken,
        int sharedExpertDim, boolean normalizeTopK,
        int linearKeyHeads, int linearValueHeads, int linearKeyDim, int linearValueDim,
        int linearConvKernel, int ngramSize, int headsPerNgram, int pleEmbedDim,
        int pleConvKernel, List<Integer> pleLayers, List<String> layerTypes,
        int indexerBudget, int indexerCompressRatio, int indexerHeadDim, int indexerHeads,
        int mtpLayers, int eosToken, List<Integer> mropeSections, boolean mropeInterleaved) implements Configuration {

    public Qwen4Configuration {
        pleLayers = List.copyOf(pleLayers);
        layerTypes = List.copyOf(layerTypes);
        mropeSections = List.copyOf(mropeSections);
        int[] positive = {dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, vocabularySize, contextLengthModel, contextLength, hcCount, hcLowrank,
                experts, expertsPerToken, sharedExpertDim, linearKeyHeads, linearValueHeads,
                linearKeyDim, linearValueDim, linearConvKernel, ngramSize, headsPerNgram,
                pleEmbedDim, pleConvKernel, indexerBudget, indexerCompressRatio, indexerHeadDim, indexerHeads};
        for (int size : positive) if (size <= 0) throw new IllegalArgumentException("Qwen4 dimensions must be positive");
        if (hcCount < 2 || expertsPerToken > experts || numberOfHeads % numberOfKeyValueHeads != 0
                || linearValueHeads % linearKeyHeads != 0 || ngramSize < 2 || mtpLayers < 0
                || contextLength > contextLengthModel || layerTypes.size() != numberOfLayers
                || rotaryDimensions <= 0 || rotaryDimensions > headSize || rotaryDimensions % 2 != 0
                || pleEmbedDim % ((ngramSize - 1) * headsPerNgram) != 0
                || !Float.isFinite(rmsNormEps) || rmsNormEps <= 0 || !Float.isFinite(ropeTheta) || ropeTheta <= 0
                || eosToken < 0 || eosToken >= vocabularySize || indexerCompressRatio<2
                || indexerBudget%indexerCompressRatio!=0 || indexerHeadDim<rotaryDimensions)
            throw new IllegalArgumentException("Inconsistent Qwen4 dimensions");
        for (String type : layerTypes) if (!type.equals("linear_attention") && !type.equals("full_attention"))
            throw new IllegalArgumentException("Unsupported Qwen4 layer type: " + type);
        if (new HashSet<>(pleLayers).size() != pleLayers.size()) throw new IllegalArgumentException("Duplicate PLE layer");
        for (int layer : pleLayers) if (layer < 1 || layer > numberOfLayers) throw new IllegalArgumentException("PLE layers are one-based");
        if(!mropeSections.isEmpty() && (mropeSections.size()!=3 || mropeSections.stream().anyMatch(n->n<=0)
                || mropeSections.stream().mapToInt(Integer::intValue).sum()!=rotaryDimensions/2))
            throw new IllegalArgumentException("Invalid MRoPE section layout");
    }

    public boolean isRecurrentLayer(int layer) { return layerTypes.get(layer).equals("linear_attention"); }
    public boolean hasPle(int layer) { return pleLayers.contains(layer + 1); }
    public int hyperDim() { return Math.multiplyExact(dim, hcCount); }
    public int linearQkvDim() { return Math.addExact(2 * linearKeyHeads * linearKeyDim, linearValueHeads * linearValueDim); }
    public int pleHeads() { return (ngramSize - 1) * headsPerNgram; }
    public int pleRowWidth() { return pleEmbedDim / pleHeads(); }
    @Override public String quantization() { return "NVFP4"; }
    @Override public DataType activationType() { return DataType.BF16; }
    @Override public int numberOfHeadsKey() { return headSize; }
    @Override public int kvDim() { return Math.multiplyExact(numberOfKeyValueHeads, headSize); }
    @Override public int kvMul() { return numberOfHeads / numberOfKeyValueHeads; }
    @Override public int keyValueLayerCount() { return (int) layerTypes.stream().filter(t -> t.equals("full_attention")).count(); }
    @Override public long recurrentStateBytes() {
        long layers = numberOfLayers - keyValueLayerCount();
        long delta = (long) linearValueHeads * linearKeyDim * linearValueDim * Float.BYTES;
        // Values are BF16-rounded, but the current host and CUDA state buffers store floats.
        long conv = (long) linearQkvDim() * (linearConvKernel - 1) * Float.BYTES;
        long ple = (long) pleLayers.size() * hyperDim() * (pleConvKernel - 1) * ngramSize * Float.BYTES;
        return Math.addExact(Math.multiplyExact(layers, delta + conv), ple);
    }

    public static Qwen4Configuration load(Path directory, int requestedContext) throws IOException {
        Map<String, Object> root = MetadataJson.parseObject(Files.readString(directory.resolve("config.json")));
        if (!"qwen4_exp".equals(root.get("model_type"))) throw new IOException("Expected qwen4_exp checkpoint");
        Map<String, Object> text = object(root.get("text_config"));
        Map<String, Object> rope = object(text.get("rope_parameters"));
        if (!"default".equals(rope.getOrDefault("rope_type", "default"))) throw new IOException("Unsupported Qwen4 RoPE scaling");
        if (!"sigmoid".equals(text.get("output_gate_type"))) throw new IOException("Unsupported Qwen4 output gate");
        if (!"silu".equals(text.get("hidden_act"))) throw new IOException("Unsupported Qwen4 MLP activation");
        if(integer(text,"indexer_kv_heads")!=1) throw new IOException("QSA indexer requires one KV head");
        int context = integer(text, "max_position_embeddings");
        int headDim = integer(text, "head_dim");
        double rotary = number(rope, "partial_rotary_factor").doubleValue() * headDim;
        if (rotary != Math.rint(rotary)) throw new IOException("Fractional RoPE dimension");
        List<Integer> ple = new ArrayList<>();
        for (Object layer : (List<?>) text.get("ple_layer_ids")) ple.add(exactInt(layer));
        List<String> types = new ArrayList<>();
        for (Object type : (List<?>) text.get("layer_types")) types.add((String) type);
        List<Integer> sections=new ArrayList<>();
        for(Object section:(List<?>)rope.getOrDefault("mrope_section",List.of())) sections.add(exactInt(section));
        return new Qwen4Configuration(integer(text,"hidden_size"), integer(text,"moe_intermediate_size"),
                integer(text,"num_hidden_layers"), integer(text,"num_attention_heads"), integer(text,"num_key_value_heads"), headDim,
                integer(text,"vocab_size"), context, requestedContext < 0 ? context : requestedContext,
                number(text,"rms_norm_eps").floatValue(), number(rope,"rope_theta").floatValue(), (int) rotary,
                integer(text,"hc_count"), integer(text,"hc_lowrank"), integer(text,"num_experts"), integer(text,"num_experts_per_tok"),
                integer(text,"shared_expert_intermediate_size"), Boolean.TRUE.equals(text.get("norm_topk_prob")),
                integer(text,"linear_num_key_heads"), integer(text,"linear_num_value_heads"), integer(text,"linear_key_head_dim"), integer(text,"linear_value_head_dim"),
                integer(text,"linear_conv_kernel_dim"), integer(text,"ngram_size"), integer(text,"heads_per_ngram"), integer(text,"ple_embed_dim"),
                integer(text,"ple_conv_kernel_size"), ple, types, integer(text,"indexer_budget"), integer(text,"indexer_compress_ratio"),
                integer(text,"indexer_head_dim"), integer(text,"indexer_n_heads"), integer(text,"mtp_num_hidden_layers"), integer(text,"eos_token_id"),
                sections,Boolean.TRUE.equals(rope.get("mrope_interleaved")));
    }

    @SuppressWarnings("unchecked") private static Map<String,Object> object(Object value) throws IOException {
        if (!(value instanceof Map)) throw new IOException("Missing Qwen4 config object");
        return (Map<String,Object>) value;
    }
    private static Number number(Map<String,Object> map, String key) throws IOException {
        if (!(map.get(key) instanceof Number value)) throw new IOException("Missing Qwen4 numeric config: " + key);
        return value;
    }
    private static int integer(Map<String,Object> map, String key) throws IOException { return exactInt(number(map,key)); }
    private static int exactInt(Object value) throws IOException {
        if (!(value instanceof Number n) || n.doubleValue() != Math.rint(n.doubleValue())
                || n.doubleValue() < Integer.MIN_VALUE || n.doubleValue() > Integer.MAX_VALUE) throw new IOException("Invalid config integer");
        return n.intValue();
    }
}

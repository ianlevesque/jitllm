package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;
import org.beehive.jitllm.server.Json;

public final class FlashNextMetadataProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextMetadataProbe MODEL_DIR REQUEST.json");
        var model=Path.of(args[0]);
        var config=Qwen4Configuration.load(model,-1);
        var tokenizer=Qwen4Tokenizer.load(model);
        var request=Json.parseObject(Files.readString(Path.of(args[1])));
        List<Object> texts=new ArrayList<>();
        for(Object value:(List<?>)request.get("texts")) {
            String text=(String)value;
            var ids=tokenizer.encode(text,tokenizer.getSpecialTokens().keySet());
            texts.add(Map.of("ids",ids,"decoded",tokenizer.decode(ids)));
        }
        Map<String,Object> dimensions=new LinkedHashMap<>();
        dimensions.put("layers",config.numberOfLayers()); dimensions.put("kv_layers",config.keyValueLayerCount());
        dimensions.put("hidden_size",config.dim()); dimensions.put("hyper_size",config.hyperDim());
        dimensions.put("kv_dim",config.kvDim()); dimensions.put("head_size",config.headSize());
        dimensions.put("rotary_dim",config.rotaryDimensions()); dimensions.put("linear_qkv_dim",config.linearQkvDim());
        dimensions.put("context",config.contextLength()); dimensions.put("ple_row_width",config.pleRowWidth());
        dimensions.put("ple_heads",config.pleHeads()); dimensions.put("ple_layers",config.pleLayers());
        dimensions.put("experts",config.experts()); dimensions.put("experts_per_token",config.expertsPerToken());
        dimensions.put("recurrent_state_bytes",config.recurrentStateBytes()); dimensions.put("mtp_layers",config.mtpLayers());
        System.out.println(Json.write(Map.of("dimensions",dimensions,"texts",texts)));
    }
}

package org.beehive.jitllm.model.format;

import java.util.List;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;

/** Flash-Next ChatML and XML tools; preserve user whitespace as the HF template does. */
public final class Qwen4ChatFormat extends Qwen35ChatFormat {
    public Qwen4ChatFormat(Qwen4Tokenizer tokenizer) {
        super(tokenizer, new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|endoftext|>", ""));
    }

    @Override public List<Integer> encodeMessage(Message message) {
        var tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeOrdinaryAsList(message.content()));
        if (!List.of(Role.FIM_PREFIX, Role.FIM_SUFFIX, Role.FIM_MIDDLE).contains(message.role())) {
            tokens.addAll(endOfTurn());
        }
        return tokens;
    }
}

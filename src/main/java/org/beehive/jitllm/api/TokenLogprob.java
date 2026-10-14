package org.beehive.jitllm.api;

import java.util.List;
import java.util.Objects;

/**
 * One generated token's log-probability and, when asked for, its most probable alternatives.
 *
 * <p>Present only on a {@link GenerationResult} (or {@link GenerationEvent}) whose {@link
 * GenerationRequest} asked for log-probabilities. The numbers come from the distribution the
 * token was actually drawn from: the temperature-scaled distribution, after any penalties and
 * before top-k / min-p / top-p truncation narrows the draw — the convention staged SGLang uses
 * by default, where the sampler reports {@code log(softmax(logits / temperature))} (and plain
 * {@code log_softmax(logits)} for greedy). The chosen token's log-probability is therefore not
 * renormalised over a truncated candidate set, and an alternative the truncation would exclude
 * can still appear in {@link #topLogprobs()}.
 *
 * @param tokenId the generated token's id, as on its {@link GenerationEvent}
 * @param text the text this token completed, exactly as on its {@link GenerationEvent} — empty
 *     while a multi-byte character is still incomplete
 * @param logprob natural log of this token's probability; {@code -Infinity} only when its
 *     probability underflowed to zero
 * @param topLogprobs up to {@link GenerationRequest#topLogprobs()} alternatives from the same
 *     distribution, most probable first; empty when none were asked for
 */
@Experimental
public record TokenLogprob(
        int tokenId, String text, double logprob, List<Alternative> topLogprobs) {

    public TokenLogprob {
        Objects.requireNonNull(text, "text; use an empty string for a token that completes none");
        topLogprobs = topLogprobs == null ? List.of() : List.copyOf(topLogprobs);
    }

    /**
     * One alternative token from the same distribution.
     *
     * @param text this token's own decoded text; empty for a token the tokenizer does not
     *     display, and possibly a replacement character for a token that carries only part of a
     *     multi-byte character on its own
     */
    public record Alternative(int tokenId, String text, double logprob) {

        public Alternative {
            Objects.requireNonNull(text, "text");
        }
    }
}

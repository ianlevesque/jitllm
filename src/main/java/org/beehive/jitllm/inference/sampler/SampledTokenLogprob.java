package org.beehive.jitllm.inference.sampler;

import java.util.List;

/**
 * One sampled token's log-probability, captured in the sampler from the distribution the token
 * was actually drawn from.
 *
 * <p>The distribution is the temperature-scaled one — {@code softmax(logits / temperature)} —
 * after any penalties the sampler applies and before top-k / min-p / top-p truncation narrows the
 * draw. That is staged SGLang's default ({@code SGLANG_RETURN_ORIGINAL_LOGPROB} off): its sampler
 * divides by temperature, softmaxes, samples, and reports {@code log(probs)} of that full
 * distribution, while the greedy path reports {@code log_softmax} of the raw logits. So a token
 * the truncation would have excluded still has a log-probability in {@link #topLogprobs()}, and
 * the chosen token's log-probability is not renormalised over the kept set.
 *
 * @param tokenId the token the sampler returned
 * @param logprob natural log of {@code tokenId}'s probability in that distribution; {@code
 *     -Infinity} only for a masked ({@code -Infinity} logit) or underflowed-to-zero probability
 * @param topLogprobs the most probable alternatives from the same distribution, most probable
 *     first (the lower token id wins a tie); empty when none were asked for
 */
public record SampledTokenLogprob(int tokenId, double logprob, List<Top> topLogprobs) {

    public SampledTokenLogprob {
        topLogprobs = topLogprobs == null ? List.of() : List.copyOf(topLogprobs);
    }

    /** One alternative token and its log-probability in the same distribution. */
    public record Top(int tokenId, double logprob) {}
}

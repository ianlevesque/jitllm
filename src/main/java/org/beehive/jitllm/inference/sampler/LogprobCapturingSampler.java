package org.beehive.jitllm.inference.sampler;

import java.util.List;
import org.beehive.jitllm.inference.Logits;

/**
 * A sampler that delegates every draw and records the drawn token's log-probability from the
 * scores the delegate leaves behind.
 *
 * <p>What the scores mean when the delegate returns is the caller's knowledge, fixed at
 * construction: the legacy temperature path scales and softmaxes its {@link Logits} in place, so
 * after the draw they hold the very probabilities sampled from ({@link Distribution#PROBABILITIES});
 * the greedy argmax sampler never touches them, so they still hold the raw logits the argmax chose
 * over ({@link Distribution#RAW_LOGITS}) and the entry is their {@code log_softmax} — the same
 * convention SGLang's greedy path uses. Nothing about the delegate's own arithmetic changes:
 * capture reads the scores after the token is chosen, so a seeded run draws the same tokens with
 * capture on or off.
 *
 * <p>The penalised/filtered path is deliberately not wrapped this way: those scores are private to
 * {@link FilteredSampler}, which captures internally instead.
 */
public final class LogprobCapturingSampler implements Sampler {

    /** What the delegate leaves in its {@link Logits} when it returns. */
    enum Distribution {
        RAW_LOGITS,
        PROBABILITIES
    }

    private final Sampler delegate;
    private final Distribution distribution;
    private final LogprobCapture capture;

    LogprobCapturingSampler(Sampler delegate, Distribution distribution, int topLogprobs) {
        this.delegate = delegate;
        this.distribution = distribution;
        this.capture = new LogprobCapture(topLogprobs);
    }

    @Override
    public int sampleToken(Logits logits) {
        int token = delegate.sampleToken(logits);
        switch (distribution) {
            case RAW_LOGITS -> capture.recordFromLogits(logits, token);
            case PROBABILITIES -> capture.recordFromProbabilities(logits, token);
        }
        return token;
    }

    @Override
    public void observeToken(int token) {
        delegate.observeToken(token);
    }

    @Override
    public List<SampledTokenLogprob> capturedLogprobs() {
        return capture.captured();
    }
}

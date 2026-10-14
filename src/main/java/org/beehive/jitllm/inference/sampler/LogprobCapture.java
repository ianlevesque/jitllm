package org.beehive.jitllm.inference.sampler;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.beehive.jitllm.inference.Logits;

/**
 * Collects one {@link SampledTokenLogprob} per {@code sampleToken} call, in call order.
 *
 * <p>The samplers feed it the scores they already have in hand — probabilities after their
 * temperature scaling and softmax, or (greedy / penalised-greedy) the logits the argmax ran over
 * — so nothing is recomputed from the model. Log-probabilities are derived in {@code double}:
 * from probabilities as {@code log(p)}, and from logits as {@code logit - logsumexp(logits)},
 * which keeps a finite logit finite where {@code log(softmax)} in {@code float} would underflow
 * to zero first.
 *
 * <p>One instance serves one generation, like the samplers that own it. Not thread-safe.
 */
final class LogprobCapture {

    /** The most alternatives any caller may ask for, matching the OpenAI {@code top_logprobs} cap. */
    static final int MAX_TOP_LOGPROBS = 20;

    private final int topCount;
    private final List<SampledTokenLogprob> captured = new ArrayList<>();

    LogprobCapture(int topCount) {
        if (topCount < 0 || topCount > MAX_TOP_LOGPROBS) {
            throw new IllegalArgumentException(
                    "topLogprobs must be in [0, " + MAX_TOP_LOGPROBS + "]: " + topCount);
        }
        this.topCount = topCount;
    }

    /** Records {@code chosen} against probabilities that already sum (about) to one. */
    void recordFromProbabilities(Logits probabilities, int chosen) {
        int n = probabilities.size();
        double[] logprobs = new double[n];
        for (int i = 0; i < n; i++) {
            logprobs[i] = Math.log(probabilities.get(i));
        }
        append(logprobs, chosen);
    }

    /** Records {@code chosen} against the first {@code n} probabilities in {@code probabilities}. */
    void recordFromProbabilities(float[] probabilities, int n, int chosen) {
        double[] logprobs = new double[n];
        for (int i = 0; i < n; i++) {
            logprobs[i] = Math.log(probabilities[i]);
        }
        append(logprobs, chosen);
    }

    /** Records {@code chosen} against raw scores, as {@code log_softmax} over them. */
    void recordFromLogits(Logits logits, int chosen) {
        int n = logits.size();
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = logits.get(i);
        }
        append(logSoftmax(values), chosen);
    }

    /** Records {@code chosen} against the first {@code n} raw scores in {@code logits}. */
    void recordFromLogits(float[] logits, int n, int chosen) {
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = logits[i];
        }
        append(logSoftmax(values), chosen);
    }

    /** Every entry recorded so far, in sample order. A live view: copy it to keep it. */
    List<SampledTokenLogprob> captured() {
        return Collections.unmodifiableList(captured);
    }

    private static double[] logSoftmax(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max) {
                max = value;
            }
        }
        double sum = 0.0;
        for (double value : values) {
            // A masked (-Infinity) logit contributes exp(-Infinity) == 0 and stays -Infinity below.
            sum += Math.exp(value - max);
        }
        double logSumExp = max + Math.log(sum);
        for (int i = 0; i < values.length; i++) {
            values[i] -= logSumExp;
        }
        return values;
    }

    private void append(double[] logprobs, int chosen) {
        captured.add(new SampledTokenLogprob(chosen, logprobs[chosen], top(logprobs)));
    }

    /** The {@code topCount} highest log-probabilities, descending; the lower id wins a tie. */
    private List<SampledTokenLogprob.Top> top(double[] logprobs) {
        int count = Math.min(topCount, logprobs.length);
        if (count == 0) {
            return List.of();
        }
        int[] best = new int[count];
        int filled = 0;
        for (int i = 0; i < logprobs.length; i++) {
            int at = filled;
            while (at > 0 && logprobs[i] > logprobs[best[at - 1]]) {
                at--;
            }
            if (at >= count) {
                continue;
            }
            int last = Math.min(filled, count - 1);
            for (int j = last; j > at; j--) {
                best[j] = best[j - 1];
            }
            best[at] = i;
            if (filled < count) {
                filled++;
            }
        }
        List<SampledTokenLogprob.Top> top = new ArrayList<>(filled);
        for (int i = 0; i < filled; i++) {
            top.add(new SampledTokenLogprob.Top(best[i], logprobs[best[i]]));
        }
        return top;
    }
}

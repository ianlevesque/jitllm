package org.beehive.jitllm.inference.sampler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;
import org.beehive.jitllm.inference.Logits;

/**
 * Sampling with repetition/frequency/presence penalties and top-k / min-p / top-p filtering.
 *
 * <p>Selected by {@link Sampler#selectSampler(int, float, float, int, float, float, float, float,
 * long)} only when at least one of those options is active, so the pre-existing greedy, categorical
 * and top-p paths keep their exact arithmetic when none is. Unlike those paths this sampler never
 * mutates the caller's {@link Logits}: penalties act on raw logits, {@code Logits} offers no
 * element write, and a private copy keeps the device view untouched.
 *
 * <p>Order per token, matching the Hugging Face processor-then-warper and vLLM order:
 *
 * <ol>
 *   <li>Repetition penalty (HF multiplicative form): for a seen token, a positive logit is divided
 *       by the penalty and a negative one is multiplied by it.
 *   <li>Frequency/presence penalties (OpenAI additive form): {@code logit -= frequencyPenalty *
 *       count + presencePenalty} for a token seen {@code count} times.
 *   <li>Temperature scaling, then softmax. Temperature 0 skips both and takes the argmax of the
 *       penalised logits (the filters below can never remove the argmax).
 *   <li>Filtering over probabilities, as the intersection of the enabled filters: top-k keeps the
 *       {@code k} highest-probability tokens, min-p keeps tokens with probability at least {@code
 *       minP} times the highest probability, and top-p keeps the smallest prefix whose cumulative
 *       probability exceeds {@code topp} (the {@link ToppSampler} convention). All three are
 *       monotone in descending probability, so the kept set is a prefix of that order and always
 *       contains the most probable token.
 *   <li>A categorical draw over the kept tokens, renormalised by their cumulative probability.
 * </ol>
 *
 * <p>"Seen" counts come from {@link #observeToken(int)} — the session seeds them with the context
 * and prompt — plus every token this sampler itself returns. One instance serves one generation
 * and is not thread-safe, like {@link ToppSampler}.
 */
public final class FilteredSampler implements Sampler {

    private final float temperature;
    private final float topp;
    private final int topk;
    private final float minp;
    private final float frequencyPenalty;
    private final float presencePenalty;
    private final float repetitionPenalty;
    private final RandomGenerator rng;

    /** Occurrences per token id; grown if a logits row or observed token exceeds it. */
    private int[] counts;

    /** Distinct seen token ids, so penalties touch only tokens that have counts. */
    private final List<Integer> seen = new ArrayList<>();

    private float[] work;
    private Integer[] order;

    /** Log-probability capture, or {@code null} when nobody asked for log-probabilities. */
    private final LogprobCapture capture;

    public FilteredSampler(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            RandomGenerator rng) {
        this(
                vocabularySize,
                temperature,
                topp,
                topk,
                minp,
                frequencyPenalty,
                presencePenalty,
                repetitionPenalty,
                rng,
                null);
    }

    /**
     * As the primary constructor, capturing each drawn token's log-probability (and up to {@code
     * topLogprobs} alternatives) from the penalised, temperature-scaled distribution this sampler
     * draws from — before its top-k / min-p / top-p truncation, like the other capture paths. At
     * temperature 0 the entry is {@code log_softmax} of the penalised logits.
     */
    FilteredSampler(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            RandomGenerator rng,
            int topLogprobs) {
        this(
                vocabularySize,
                temperature,
                topp,
                topk,
                minp,
                frequencyPenalty,
                presencePenalty,
                repetitionPenalty,
                rng,
                new LogprobCapture(topLogprobs));
    }

    private FilteredSampler(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            RandomGenerator rng,
            LogprobCapture capture) {
        if (vocabularySize <= 0) {
            throw new IllegalArgumentException(
                    "vocabularySize must be positive: " + vocabularySize);
        }
        // Written so NaN, which fails every comparison, is rejected rather than silently kept.
        if (!(temperature >= 0)) {
            throw new IllegalArgumentException("temperature must be 0 or greater: " + temperature);
        }
        if (!(minp >= 0 && minp <= 1)) {
            throw new IllegalArgumentException("minP must be in [0, 1]: " + minp);
        }
        if (!Float.isFinite(frequencyPenalty)) {
            throw new IllegalArgumentException(
                    "frequencyPenalty must be finite: " + frequencyPenalty);
        }
        if (!Float.isFinite(presencePenalty)) {
            throw new IllegalArgumentException(
                    "presencePenalty must be finite: " + presencePenalty);
        }
        if (!(repetitionPenalty > 0) || !Float.isFinite(repetitionPenalty)) {
            throw new IllegalArgumentException(
                    "repetitionPenalty must be positive and finite: " + repetitionPenalty);
        }
        this.temperature = temperature;
        this.topp = topp;
        this.topk = topk;
        this.minp = minp;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        this.repetitionPenalty = repetitionPenalty;
        this.rng = rng;
        this.capture = capture;
        this.counts = new int[vocabularySize];
        this.work = new float[vocabularySize];
        this.order = new Integer[vocabularySize];
    }

    @Override
    public void observeToken(int token) {
        if (token < 0) {
            throw new IllegalArgumentException("token must be non-negative: " + token);
        }
        ensureCapacity(token + 1);
        if (counts[token]++ == 0) {
            seen.add(token);
        }
    }

    /** How often {@code token} has been observed or sampled. */
    int tokenCount(int token) {
        return token >= 0 && token < counts.length ? counts[token] : 0;
    }

    @Override
    public int sampleToken(Logits logits) {
        int n = logits.size();
        ensureCapacity(n);
        for (int i = 0; i < n; i++) {
            work[i] = logits.get(i);
        }
        applyPenalties(n);

        int token = temperature == 0.0f ? argmax(n) : draw(n);
        if (capture != null) {
            // At this point work holds exactly what was drawn from: the penalised logits on the
            // greedy path, or their temperature-scaled softmax (draw softmaxes work in place and
            // only reorders the separate order array).
            if (temperature == 0.0f) {
                capture.recordFromLogits(work, n, token);
            } else {
                capture.recordFromProbabilities(work, n, token);
            }
        }
        observeToken(token);
        return token;
    }

    @Override
    public List<SampledTokenLogprob> capturedLogprobs() {
        return capture == null ? List.of() : capture.captured();
    }

    private void applyPenalties(int n) {
        boolean multiplicative = repetitionPenalty != 1.0f;
        boolean additive = frequencyPenalty != 0.0f || presencePenalty != 0.0f;
        if (!multiplicative && !additive) {
            return;
        }
        for (int token : seen) {
            if (token >= n) {
                continue;
            }
            float logit = work[token];
            if (multiplicative) {
                logit = logit < 0 ? logit * repetitionPenalty : logit / repetitionPenalty;
            }
            if (additive) {
                logit -= frequencyPenalty * counts[token] + presencePenalty;
            }
            work[token] = logit;
        }
    }

    private int argmax(int n) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (work[i] > bestValue) {
                bestValue = work[i];
                best = i;
            }
        }
        return best;
    }

    private int draw(int n) {
        softmaxAfterTemperature(n);

        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        // Descending probability; the lower id wins a tie, as the argmax path keeps the first max.
        Arrays.sort(
                order,
                0,
                n,
                (a, b) -> {
                    int byProbability = Float.compare(work[b], work[a]);
                    return byProbability != 0 ? byProbability : Integer.compare(a, b);
                });

        float minThreshold = minp > 0 ? minp * work[order[0]] : 0.0f;
        boolean nucleus = topp > 0 && topp < 1;
        float cumulative = 0.0f;
        int kept = 0;
        for (int rank = 0; rank < n; rank++) {
            if (rank > 0) {
                if (topk > 0 && rank >= topk) {
                    break;
                }
                if (work[order[rank]] < minThreshold) {
                    break;
                }
                if (nucleus && cumulative > topp) {
                    break;
                }
            }
            cumulative += work[order[rank]];
            kept++;
        }

        float r = rng.nextFloat(1f) * cumulative;
        float cdf = 0.0f;
        for (int rank = 0; rank < kept; rank++) {
            cdf += work[order[rank]];
            if (r < cdf) {
                return order[rank];
            }
        }
        return order[kept - 1]; // in case of rounding errors
    }

    private void softmaxAfterTemperature(int n) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            work[i] /= temperature;
            if (work[i] > max) {
                max = work[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // A masked (-inf) logit stays a zero probability, as exp(-inf) is 0.
            work[i] = (float) Math.exp(work[i] - max);
            sum += work[i];
        }
        for (int i = 0; i < n; i++) {
            work[i] /= sum;
        }
    }

    private void ensureCapacity(int size) {
        if (size <= counts.length) {
            return;
        }
        counts = Arrays.copyOf(counts, size);
        work = Arrays.copyOf(work, size);
        order = Arrays.copyOf(order, size);
    }
}

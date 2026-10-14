package org.beehive.jitllm.inference.sampler;

import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.beehive.jitllm.Options;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Generic interface for sampling tokens from probability distributions. Supports both FloatTensor
 * and FloatArray tensor implementations.
 */
@FunctionalInterface
public interface Sampler {

    /** Argmax implementation for FloatTensor. */
    Sampler TENSOR_ARGMAX =
            tensor -> {
                int maxIndex = 0;
                float maxValue = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < tensor.size(); i++) {
                    float value = tensor.get(i);
                    if (value > maxValue) {
                        maxValue = value;
                        maxIndex = i;
                    }
                }
                return maxIndex;
            };

    /**
     * Legacy ARGMAX for backward compatibility.
     *
     * @deprecated Use TENSOR_ARGMAX instead
     */
    @Deprecated Sampler ARGMAX = TENSOR_ARGMAX;

    /**
     * Creates and configures a sampler for token generation based on specified parameters.
     *
     * <p>This method selects an appropriate sampling strategy for next-token prediction in language
     * model inference. It supports several sampling approaches:
     *
     * <ul>
     *   <li>Greedy sampling (temperature = 0): Always selects the most probable token
     *   <li>Temperature sampling: Adjusts probability distribution sharpness
     *   <li>Top-p (nucleus) sampling: Considers only tokens comprising the top p probability mass
     * </ul>
     *
     * <p>The method handles both {@link FloatTensor} and {@link FloatArray} logits types to support
     * both CPU and GPU execution paths.
     *
     * @param vocabularySize The size of the model's vocabulary
     * @param temperature A value controlling randomness in sampling:
     *     <ul>
     *       <li>0.0f: No randomness (greedy sampling)
     *       <li>1.0f: Standard sampling from unmodified distribution
     *       <li>&lt;1.0f: More deterministic (sharper distribution)
     *       <li>&gt;1.0f: More random (flatter distribution)
     *     </ul>
     *
     * @param topp The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *     <ul>
     *       <li>Values ≤0 or ≥1: Disables top-p sampling
     *       <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass
     *     </ul>
     *
     * @param rngSeed Seed value for the random number generator to ensure reproducibility
     * @return A configured {@link Sampler} that implements the selected sampling strategy and
     *     handles both tensor and array-based logits
     * @throws IllegalArgumentException if logits are of an unsupported type
     */
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        return selectSampler(vocabularySize, temperature, topp, 0, 0.0f, 0.0f, 0.0f, 1.0f, rngSeed);
    }

    /**
     * {@link #selectSampler(int, float, float, int, float, float, float, float, long)} capturing
     * log-probabilities: the returned sampler's {@link #capturedLogprobs()} gains one entry per
     * {@link #sampleToken(Logits)} call, including up to {@code topLogprobs} alternatives ({@code
     * 0} for none), from the distribution actually sampled — see {@link SampledTokenLogprob}.
     * Capture only reads scores after each draw, so the tokens drawn for a seed are exactly those
     * the capturing-free factory draws.
     *
     * @param topLogprobs alternatives to keep per token, in {@code [0, 20]}
     */
    static Sampler selectSampler(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            long rngSeed,
            int topLogprobs) {
        boolean filtered =
                topk > 0
                        || minp > 0
                        || frequencyPenalty != 0.0f
                        || presencePenalty != 0.0f
                        || repetitionPenalty != 1.0f;
        if (filtered) {
            return new FilteredSampler(
                    vocabularySize,
                    temperature,
                    topp,
                    topk,
                    minp,
                    frequencyPenalty,
                    presencePenalty,
                    repetitionPenalty,
                    RandomGeneratorFactory.getDefault().create(rngSeed),
                    topLogprobs);
        }
        Sampler delegate =
                selectSampler(
                        vocabularySize,
                        temperature,
                        topp,
                        topk,
                        minp,
                        frequencyPenalty,
                        presencePenalty,
                        repetitionPenalty,
                        rngSeed);
        // The greedy delegate leaves raw logits untouched; the temperature delegate leaves the
        // probabilities it sampled from in their place. The wrapper reads whichever it is.
        return new LogprobCapturingSampler(
                delegate,
                temperature == 0.0f
                        ? LogprobCapturingSampler.Distribution.RAW_LOGITS
                        : LogprobCapturingSampler.Distribution.PROBABILITIES,
                topLogprobs);
    }

    /**
     * {@link #selectSampler(int, float, float, long)} with top-k, min-p and penalty controls.
     *
     * <p>When every added control is at its disabled value ({@code topk <= 0}, {@code minp <= 0},
     * both additive penalties {@code 0}, {@code repetitionPenalty == 1}) the sampler is exactly the
     * one the four-argument factory has always returned, arithmetic included. Otherwise a {@link
     * FilteredSampler} applies the controls; see its javadoc for the order and semantics.
     *
     * @param topk Keep only the {@code k} highest-probability tokens; {@code 0} or negative (some
     *     APIs send {@code -1}) disables it.
     * @param minp Keep only tokens whose probability is at least {@code minp} times the highest
     *     probability, in {@code [0, 1]}; {@code 0} disables it.
     * @param frequencyPenalty OpenAI additive penalty per previous occurrence of a token.
     * @param presencePenalty OpenAI additive penalty applied once per previously seen token.
     * @param repetitionPenalty Hugging Face multiplicative penalty; {@code 1} disables it.
     */
    static Sampler selectSampler(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            long rngSeed) {
        boolean filtered =
                topk > 0
                        || minp > 0
                        || frequencyPenalty != 0.0f
                        || presencePenalty != 0.0f
                        || repetitionPenalty != 1.0f;
        if (filtered) {
            return new FilteredSampler(
                    vocabularySize,
                    temperature,
                    topp,
                    topk,
                    minp,
                    frequencyPenalty,
                    presencePenalty,
                    repetitionPenalty,
                    RandomGeneratorFactory.getDefault().create(rngSeed));
        }
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            // Determine whether to use top-p (nucleus) sampling
            if (topp <= 0 || topp >= 1) {
                // If topp is outside (0,1), use standard categorical sampling
                // This samples directly from the probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // Use top-p (nucleus) sampling with the specified threshold
                // This restricts sampling to only the most likely tokens that
                // cumulatively comprise the top p probability mass
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }

            // Create a sampler that:
            // 1. Applies temperature scaling to the logits
            // 2. Converts logits to probabilities using softmax
            // 3. Delegates the actual sampling to the appropriate inner sampler
            sampler =
                    logits -> {
                        // Temperature scaling, then softmax. Which implementation runs is the
                        // view's business: the host and device paths keep the code they had, so the
                        // arithmetic and its ordering are unchanged.
                        logits.divideInPlace(0, logits.size(), temperature);
                        logits.softmaxInPlace(0, logits.size());
                        return innerSampler.sampleToken(logits);
                    };
        }
        return sampler;
    }

    static Sampler createSampler(Model model, Options options) {
        float temperature =
                Float.isNaN(options.temperature())
                        ? (float) model.chatFormat().defaultTemperature()
                        : options.temperature();
        float topp =
                Float.isNaN(options.topp())
                        ? (float) model.chatFormat().defaultTopP()
                        : options.topp();
        return selectSampler(
                model.configuration().vocabularySize(), temperature, topp, options.seed());
    }

    /**
     * Find the index of the maximum value in a FloatArray.
     *
     * @param array The FloatArray to find the maximum value in
     * @return The index of the maximum value
     */

    /**
     * Sample a token from the provided tensor.
     *
     * @param tensor The tensor containing probabilities/logits
     * @return The selected token index
     */
    int sampleToken(Logits logits);

    /**
     * The log-probability entries captured so far, one per {@link #sampleToken(Logits)} call in
     * call order. Empty unless this sampler was created by a capturing factory: most samplers
     * capture nothing, and a generation that never asked for log-probabilities should not pay for
     * computing them.
     */
    default List<SampledTokenLogprob> capturedLogprobs() {
        return List.of();
    }

    /**
     * Records a token as part of the sequence so far, for samplers whose penalties depend on
     * history. The default does nothing: most samplers are history-free, and a penalty sampler
     * already records every token it returns, so this is how a caller seeds it with the tokens it
     * did not sample — the retained context and the prompt.
     */
    default void observeToken(int token) {}

    /** Records a stretch of history; see {@link #observeToken(int)}. */
    default void observeTokens(List<Integer> tokens) {
        for (int token : tokens) {
            observeToken(token);
        }
    }
}

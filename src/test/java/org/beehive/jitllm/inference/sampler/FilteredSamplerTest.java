package org.beehive.jitllm.inference.sampler;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.junit.Test;

/**
 * Semantics of top-k, min-p and the three penalties, exercised through the {@link Sampler} factory
 * the session uses. Greedy (temperature 0) cases pin the penalty arithmetic exactly; sampling cases
 * pin which tokens can ever be drawn.
 */
public class FilteredSamplerTest {

    /** Fresh logits per draw: the legacy sampling path scales and softmaxes its input in place. */
    private static Logits logits(float... values) {
        return Logits.of(new ArrayFloatTensor(values.clone()));
    }

    private static Sampler filtered(
            int vocabularySize,
            float temperature,
            float topp,
            int topk,
            float minp,
            float frequencyPenalty,
            float presencePenalty,
            float repetitionPenalty,
            long seed) {
        return Sampler.selectSampler(
                vocabularySize,
                temperature,
                topp,
                topk,
                minp,
                frequencyPenalty,
                presencePenalty,
                repetitionPenalty,
                seed);
    }

    private static List<Integer> drawRepeatedly(Sampler sampler, float[] values, int draws) {
        List<Integer> tokens = new ArrayList<>();
        for (int i = 0; i < draws; i++) {
            tokens.add(sampler.sampleToken(logits(values)));
        }
        return tokens;
    }

    @Test
    public void disabledControlsReproduceTheLegacySamplerExactly() {
        float[] values = {0.4f, 1.7f, -0.6f, 2.9f, 0.1f, -2.0f, 1.1f, 0.8f};
        for (int disabledTopK : new int[] {0, -1}) {
            Sampler legacy = Sampler.selectSampler(values.length, 0.9f, 0.8f, 42L);
            Sampler extended =
                    filtered(values.length, 0.9f, 0.8f, disabledTopK, 0f, 0f, 0f, 1f, 42L);
            assertEquals(
                    "topK=" + disabledTopK + " must not alter a single draw",
                    drawRepeatedly(legacy, values, 40),
                    drawRepeatedly(extended, values, 40));
        }
    }

    @Test
    public void topKOneAlwaysPicksTheArgmax() {
        float[] values = {0f, 3f, 1f, 2f};
        Sampler sampler = filtered(values.length, 1f, 1f, 1, 0f, 0f, 0f, 1f, 7L);
        for (int token : drawRepeatedly(sampler, values, 50)) {
            assertEquals(1, token);
        }
    }

    @Test
    public void topKLimitsTheCandidateSet() {
        float[] values = {3f, 2.9f, 0f, -5f};
        Sampler sampler = filtered(values.length, 1f, 1f, 2, 0f, 0f, 0f, 1f, 11L);
        Set<Integer> drawn = new HashSet<>(drawRepeatedly(sampler, values, 200));
        assertEquals("both kept tokens are reachable, and nothing else", Set.of(0, 1), drawn);
    }

    @Test
    public void minPKeepsOnlyTokensNearTheMaximumProbability() {
        // Ratios to the top probability: exp(-0.1) ~ 0.90, exp(-4) ~ 0.02, exp(-5) ~ 0.007.
        float[] values = {1f, 0.9f, -3f, -4f};
        Sampler half = filtered(values.length, 1f, 1f, 0, 0.5f, 0f, 0f, 1f, 13L);
        assertEquals(Set.of(0, 1), new HashSet<>(drawRepeatedly(half, values, 200)));

        Sampler strict = filtered(values.length, 1f, 1f, 0, 0.99f, 0f, 0f, 1f, 13L);
        for (int token : drawRepeatedly(strict, values, 50)) {
            assertEquals("min_p above the runner-up's ratio keeps only the argmax", 0, token);
        }
    }

    @Test
    public void frequencyPenaltyIsChargedPerOccurrenceIncludingSelfSampledTokens() {
        float[] values = {1f, 0.9f, 0f};
        Sampler sampler = filtered(values.length, 0f, 1f, 0, 0f, 0.4f, 0f, 1f, 1L);
        assertEquals(0, sampler.sampleToken(logits(values)));
        assertEquals(
                "the just-sampled token now counts once: 1.0 - 0.4 < 0.9",
                1,
                sampler.sampleToken(logits(values)));
    }

    @Test
    public void presencePenaltyIsChargedOncePerSeenToken() {
        float[] values = {1f, 0.8f, 0.75f};

        Sampler presence = filtered(values.length, 0f, 1f, 0, 0f, 0f, 0.3f, 1f, 1L);
        presence.observeToken(0);
        assertEquals(
                "one sighting already costs the full 0.3", 1, presence.sampleToken(logits(values)));

        Sampler presenceRepeated = filtered(values.length, 0f, 1f, 0, 0f, 0f, 0.3f, 1f, 1L);
        presenceRepeated.observeTokens(List.of(0, 0, 0, 0, 0));
        assertEquals(
                "further sightings add nothing", 1, presenceRepeated.sampleToken(logits(values)));

        Sampler frequencyOnce = filtered(values.length, 0f, 1f, 0, 0f, 0.06f, 0f, 1f, 1L);
        frequencyOnce.observeToken(0);
        assertEquals(
                "one frequency charge (0.94) still wins",
                0,
                frequencyOnce.sampleToken(logits(values)));

        Sampler frequencyFiveTimes = filtered(values.length, 0f, 1f, 0, 0f, 0.06f, 0f, 1f, 1L);
        frequencyFiveTimes.observeTokens(List.of(0, 0, 0, 0, 0));
        assertEquals(
                "five frequency charges (0.70) lose",
                1,
                frequencyFiveTimes.sampleToken(logits(values)));
    }

    @Test
    public void repetitionPenaltyDividesPositiveLogitsAndMultipliesNegativeOnes() {
        float[] positive = {2f, 1.5f, 0f};
        Sampler positiveCase = filtered(positive.length, 0f, 1f, 0, 0f, 0f, 0f, 2f, 1L);
        assertEquals(0, positiveCase.sampleToken(logits(positive)));
        assertEquals(
                "seen positive logit 2.0 / 2 = 1.0 now trails 1.5",
                1,
                positiveCase.sampleToken(logits(positive)));

        float[] negative = {-1f, -1.9f, -6f};
        Sampler negativeCase = filtered(negative.length, 0f, 1f, 0, 0f, 0f, 0f, 2f, 1L);
        negativeCase.observeToken(0);
        assertEquals(
                "seen negative logit -1.0 * 2 = -2.0 now trails -1.9",
                1,
                negativeCase.sampleToken(logits(negative)));
    }

    @Test
    public void combinedControlsAreDeterministicForASeed() {
        float[] values = {2.2f, 1.9f, 1.4f, 0.3f, -0.8f, 2.0f};
        List<List<Integer>> runs = new ArrayList<>();
        for (int run = 0; run < 2; run++) {
            Sampler sampler = filtered(values.length, 0.8f, 0.9f, 3, 0.05f, 0.1f, 0.05f, 1.1f, 99L);
            sampler.observeTokens(List.of(0, 5, 5));
            runs.add(drawRepeatedly(sampler, values, 30));
        }
        assertEquals(runs.get(0), runs.get(1));
    }

    @Test
    public void filteredSamplingLeavesTheCallersLogitsUntouched() {
        float[] values = {1f, 0.9f, 0f};
        ArrayFloatTensor tensor = new ArrayFloatTensor(values.clone());
        Sampler sampler = filtered(values.length, 0.7f, 0.9f, 2, 0.1f, 0.5f, 0.25f, 1.2f, 3L);
        sampler.observeToken(0);
        sampler.sampleToken(Logits.of(tensor));
        for (int i = 0; i < values.length; i++) {
            assertEquals(values[i], tensor.getFloat(i), 0.0f);
        }
    }

    @Test
    public void maskedNegativeInfinityLogitsNeverWin() {
        float[] values = {0.5f, Float.NEGATIVE_INFINITY, 0.4f, Float.NEGATIVE_INFINITY};
        Sampler sampler = filtered(values.length, 1f, 1f, 0, 0f, 0.2f, 0f, 1.5f, 5L);
        sampler.observeTokens(List.of(1, 3));
        for (int token : drawRepeatedly(sampler, values, 50)) {
            assertTrue("masked tokens stay unreachable", token == 0 || token == 2);
        }
    }

    @Test
    public void invalidControlValuesAreRejected() {
        var rng = java.util.random.RandomGeneratorFactory.getDefault().create(1L);
        assertThrows(
                "repetition penalty 0 has no multiplicative meaning",
                IllegalArgumentException.class,
                () -> new FilteredSampler(4, 1f, 1f, 0, 0f, 0f, 0f, 0f, rng));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FilteredSampler(4, 1f, 1f, 0, 1.5f, 0f, 0f, 1f, rng));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FilteredSampler(4, 1f, 1f, 0, 0f, Float.NaN, 0f, 1f, rng));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FilteredSampler(4, -1f, 1f, 0, 0f, 0f, 0f, 1f, rng));
    }
}

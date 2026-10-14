package org.beehive.jitllm.inference.sampler;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.junit.Test;

/**
 * Log-probability capture against hand-computed softmaxes. Expected values are computed here in
 * {@code double} straight from the known logits — the distribution actually sampled:
 * temperature-scaled, penalties applied, truncation not renormalised.
 */
public class LogprobCaptureTest {

    private static Logits logits(float... values) {
        return Logits.of(new ArrayFloatTensor(values.clone()));
    }

    /** log_softmax(logits / temperature)[index], computed independently of the sampler. */
    private static double expectedLogprob(float[] values, float temperature, int index) {
        double max = Double.NEGATIVE_INFINITY;
        for (float value : values) {
            max = Math.max(max, value / (double) temperature);
        }
        double sum = 0;
        for (float value : values) {
            sum += Math.exp(value / (double) temperature - max);
        }
        return values[index] / (double) temperature - (max + Math.log(sum));
    }

    @Test
    public void greedyCaptureIsLogSoftmaxOfTheRawLogits() {
        float[] values = {2f, 1f, 0f, -1f};
        Sampler sampler = Sampler.selectSampler(values.length, 0f, 1f, 0, 0f, 0f, 0f, 1f, 7L, 2);
        int token = sampler.sampleToken(logits(values));
        assertEquals("greedy draw is unchanged by capture", 0, token);

        var captured = sampler.capturedLogprobs();
        assertEquals(1, captured.size());
        var entry = captured.get(0);
        assertEquals(0, entry.tokenId());
        assertEquals(
                "2 - log(e^2 + e + 1 + e^-1)",
                expectedLogprob(values, 1f, 0),
                entry.logprob(),
                1e-9);
        assertEquals(2, entry.topLogprobs().size());
        assertEquals(0, entry.topLogprobs().get(0).tokenId());
        assertEquals(1, entry.topLogprobs().get(1).tokenId());
        assertEquals(
                expectedLogprob(values, 1f, 1), entry.topLogprobs().get(1).logprob(), 1e-9);
    }

    @Test
    public void temperatureCaptureMatchesTheScaledDistributionActuallySampled() {
        float[] values = {2f, 1f, 0f, -1f, 0.5f};
        Sampler sampler =
                Sampler.selectSampler(values.length, 2f, 1f, 0, 0f, 0f, 0f, 1f, 42L, 3);
        for (int draw = 0; draw < 25; draw++) {
            int token = sampler.sampleToken(logits(values));
            var entry = sampler.capturedLogprobs().get(draw);
            assertEquals(token, entry.tokenId());
            // The legacy path softmaxes in float, so log(p) agrees only to float precision.
            assertEquals(expectedLogprob(values, 2f, token), entry.logprob(), 1e-5);
            assertEquals(3, entry.topLogprobs().size());
            assertEquals(
                    "the most probable alternative is the argmax whatever was drawn",
                    0,
                    entry.topLogprobs().get(0).tokenId());
        }
        assertEquals(25, sampler.capturedLogprobs().size());
    }

    @Test
    public void capturingNeverChangesTheDrawsForASeed() {
        float[] values = {0.4f, 1.7f, -0.6f, 2.9f, 0.1f, -2.0f, 1.1f, 0.8f};
        for (float topp : new float[] {1f, 0.8f}) {
            Sampler plain = Sampler.selectSampler(values.length, 0.9f, topp, 123L);
            Sampler capturing =
                    Sampler.selectSampler(values.length, 0.9f, topp, 0, 0f, 0f, 0f, 1f, 123L, 5);
            List<Integer> plainDraws = new ArrayList<>();
            List<Integer> capturingDraws = new ArrayList<>();
            for (int i = 0; i < 60; i++) {
                plainDraws.add(plain.sampleToken(logits(values)));
                capturingDraws.add(capturing.sampleToken(logits(values)));
            }
            assertEquals("topp=" + topp, plainDraws, capturingDraws);
            assertTrue(plain.capturedLogprobs().isEmpty());
            assertEquals(60, capturing.capturedLogprobs().size());
        }
    }

    @Test
    public void filteredCaptureIncludesPenaltiesButNotTruncationRenormalisation() {
        float[] values = {2f, 1f, 0f};
        // topK=1 forces token 0; token 1 is nevertheless reported with its full-distribution
        // probability, after the frequency penalty on the once-observed token 0.
        Sampler sampler =
                Sampler.selectSampler(values.length, 1f, 1f, 1, 0f, 0.5f, 0f, 1f, 3L, 3);
        sampler.observeToken(0);
        int token = sampler.sampleToken(logits(values));
        assertEquals(0, token);

        float[] penalised = {2f - 0.5f, 1f, 0f};
        var entry = sampler.capturedLogprobs().get(0);
        assertEquals(expectedLogprob(penalised, 1f, 0), entry.logprob(), 1e-5);
        assertEquals(1, entry.topLogprobs().get(1).tokenId());
        assertEquals(
                "token 1 keeps its un-truncated probability",
                expectedLogprob(penalised, 1f, 1),
                entry.topLogprobs().get(1).logprob(),
                1e-5);
    }

    @Test
    public void topEntriesBreakTiesByLowerTokenIdAndCapAtVocabulary() {
        float[] values = {1f, 1f, 1f};
        Sampler sampler = Sampler.selectSampler(values.length, 0f, 1f, 0, 0f, 0f, 0f, 1f, 1L, 20);
        sampler.sampleToken(logits(values));
        var tops = sampler.capturedLogprobs().get(0).topLogprobs();
        assertEquals("asked for 20 of 3", 3, tops.size());
        assertEquals(0, tops.get(0).tokenId());
        assertEquals(1, tops.get(1).tokenId());
        assertEquals(2, tops.get(2).tokenId());
        assertEquals(-Math.log(3), tops.get(0).logprob(), 1e-9);
    }

    @Test
    public void invalidTopCountsAreRejected() {
        try {
            Sampler.selectSampler(4, 1f, 1f, 0, 0f, 0f, 0f, 1f, 1L, 21);
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("topLogprobs 21 must be rejected");
    }
}

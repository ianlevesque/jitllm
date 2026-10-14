package org.beehive.jitllm.server;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

public final class InferenceServiceMetricsTest {

    @Before
    public void resetCounters() {
        InferenceService.Metrics.reset();
    }

    @Test
    public void renderCarriesHelpTypeAndValues() {
        InferenceService.Metrics.recordRequest("/v1/chat/completions");
        InferenceService.Metrics.recordRequest("/v1/chat/completions");
        InferenceService.Metrics.recordRequest("/v1/models");

        InferenceService.Metrics.beginGeneration();
        assertTrue(InferenceService.Metrics.render().contains("jitllm_running_requests 1\n"));
        InferenceService.Metrics.finishGeneration(1_500_000_000L, 10, 5);

        String out = InferenceService.Metrics.render();
        assertTrue(out, out.contains("# HELP jitllm_requests_total "));
        assertTrue(out, out.contains("# TYPE jitllm_requests_total counter\n"));
        assertTrue(out, out.contains("jitllm_requests_total{endpoint=\"/v1/chat/completions\"} 2\n"));
        assertTrue(out, out.contains("jitllm_requests_total{endpoint=\"/v1/models\"} 1\n"));
        assertTrue(out, out.contains("# TYPE jitllm_generations_total counter\n"));
        assertTrue(out, out.contains("jitllm_generations_total 1\n"));
        assertTrue(out, out.contains("jitllm_prompt_tokens_total 10\n"));
        assertTrue(out, out.contains("jitllm_completion_tokens_total 5\n"));
        assertTrue(out, out.contains("# TYPE jitllm_generation_duration_seconds summary\n"));
        assertTrue(out, out.contains("jitllm_generation_duration_seconds_sum 1.5\n"));
        assertTrue(out, out.contains("jitllm_generation_duration_seconds_count 1\n"));
        assertTrue(out, out.contains("# TYPE jitllm_running_requests gauge\n"));
        assertTrue(out, out.contains("jitllm_running_requests 0\n"));
    }

    @Test
    public void countersAccumulateAcrossGenerations() {
        InferenceService.Metrics.beginGeneration();
        InferenceService.Metrics.finishGeneration(1_000_000_000L, 3, 2);
        InferenceService.Metrics.beginGeneration();
        InferenceService.Metrics.finishGeneration(2_000_000_000L, 4, 6);

        String out = InferenceService.Metrics.render();
        assertTrue(out, out.contains("jitllm_generations_total 2\n"));
        assertTrue(out, out.contains("jitllm_prompt_tokens_total 7\n"));
        assertTrue(out, out.contains("jitllm_completion_tokens_total 8\n"));
        assertTrue(out, out.contains("jitllm_generation_duration_seconds_sum 3.0\n"));
        assertTrue(out, out.contains("jitllm_running_requests 0\n"));
    }
}

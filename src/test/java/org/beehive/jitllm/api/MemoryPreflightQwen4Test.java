package org.beehive.jitllm.api;

import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.junit.Test;

/** The Flash-Next preflight arithmetic at the checkpoint's real dimensions (262144 native). */
public final class MemoryPreflightQwen4Test {

    /** nvidia/Qwen3.8-Flash-Next-NVFP4 text_config, reduced to the fields the math reads. */
    private static Qwen4Configuration flashNext(int context) {
        var types = new ArrayList<String>();
        for (int i = 0; i < 48; i++) types.add(i % 4 == 3 ? "full_attention" : "linear_attention");
        return new Qwen4Configuration(2560, 1024, 48, 24, 2, 256, 248320, 262144, context,
                1e-6f, 10000000f, 64, 4, 8, 512, 10, 512, true,
                16, 48, 128, 128, 4, 3, 8, 2560, 4, List.of(2), types,
                2048, 4, 128, 4, 1, 248046, List.of(11, 11, 10), true);
    }

    @Test public void pagedKvMatchesExecutorPageGeometryAtFullContext() {
        var config = flashNext(262144);
        assertEquals(2048, MemoryPreflight.qwen4QsaPageTokens(config));
        assertEquals(12, config.keyValueLayerCount());
        // 128 pages/layer: FP8 cache 2048*2*512 B + BF16 compressed 512*128*2 B each, plus a
        // BF16 fresh shadow (2 B/element) on only the two pages inside the 4096-token window.
        long perPage = 2048L * 2 * 512 + 512L * 128 * 2;
        long expectedPerLayer = 128 * perPage + 2 * (2048L * 2 * 512 * 2) + 126 * 2L;
        assertEquals(expectedPerLayer, MemoryPreflight.qwen4PagedKvBytesPerLayerSession(config));
        assertEquals(293601532L, MemoryPreflight.qwen4PagedKvBytesPerLayerSession(config));
        assertEquals(3523218384L, MemoryPreflight.qwen4PagedKvBytesPerSession(config));
    }

    @Test public void pagedKvScalesWithContextAndKeepsTheFreshWindow() {
        var small = flashNext(4096);
        // Both pages sit inside the fresh window, so both carry the BF16 shadow.
        assertEquals(2 * (2048L * 2 * 512 + 512L * 128 * 2 + 2048L * 2 * 512 * 2),
                MemoryPreflight.qwen4PagedKvBytesPerLayerSession(small));
        assertTrue(MemoryPreflight.qwen4PagedKvBytesPerSession(flashNext(262144))
                > MemoryPreflight.qwen4PagedKvBytesPerSession(flashNext(131072)));
    }

    @Test public void recurrentStateCountsCheckpointBanksNotOneState() {
        var config = flashNext(262144);
        long banked = MemoryPreflight.qwen4RecurrentBytesPerSession(config, 8);
        // 36 layers x (state + 8 banks) is ~1 GiB, where the single-state figure is ~113 MiB.
        assertEquals(1071535104L, banked);
        assertTrue(banked > 8 * config.recurrentStateBytes());
        assertEquals(0, MemoryPreflight.qwen4RecurrentBytesPerSession(config.mtpConfiguration(), 8));
    }

    @Test public void threeFullContextSessionsFitEveryDefaultExecutorBudget() {
        var config = flashNext(262144);
        assertNull(MemoryPreflight.qwen4ExecutorCapViolation(config, 3));
        long qsa = 3 * (MemoryPreflight.qwen4PagedKvBytesPerSession(config)
                + MemoryPreflight.qwen4QsaFixedBytesPerSession(config));
        assertTrue(qsa < 16L << 30);
        assertEquals(29848572L, MemoryPreflight.qwen4QsaFixedBytesPerLayerSession(config));
    }

    @Test public void sessionCountPastTheBudgetsIsRefusedWithTheBindingCap() {
        var config = flashNext(262144);
        String violation = MemoryPreflight.qwen4ExecutorCapViolation(config, 8);
        assertNotNull(violation);
        assertTrue(violation, violation.contains("jitllm.qwen4.gpuQsaBytes"));
        // A short context admits sessions the full context cannot: the cap is arithmetic, not a
        // blanket session limit.
        assertNull(MemoryPreflight.qwen4ExecutorCapViolation(flashNext(4096), 3));
    }

    @Test public void planChargesEveryAdmittedSessionAndStaysConservative() {
        var config = flashNext(262144);
        MemoryPlan plan = MemoryPreflight.qwen4Plan(config,
                ModelOptions.builder().contextLength(262144).maxConcurrentSessions(3).build());
        assertEquals(MemoryPlan.Confidence.CONSERVATIVE, plan.confidence());
        long kv = plan.components().stream()
                .filter(c -> c.name().startsWith("FP8 paged KV"))
                .mapToLong(c -> c.logicalBytes()).findFirst().orElseThrow();
        assertEquals(3 * 3523218384L, kv);
        assertTrue(plan.assumptions(), plan.assumptions().contains("262144"));
        assertFalse(plan.assumptions().contains("CANNOT FIT"));
        // FP8, not the generic model's FP16/FP32 charge: under the 95 GiB card even with the
        // default 16 GiB weight-cache budget included.
        assertTrue(plan.predictedBudgetBytes() < 95L << 30);
    }
}

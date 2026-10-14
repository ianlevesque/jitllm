package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.inference.sampler.SampledTokenLogprob;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.junit.Test;

/** Per-request thinking/logprob controls: request surface, encoder resolution, event pairing. */
public class GenerationRequestControlsTest {

    @Test
    public void defaultsInheritTheSessionAndCaptureNothing() {
        GenerationRequest request = GenerationRequest.of("x");
        assertNull("unset thinking means the session mode", request.thinkingMode());
        assertNull(request.reasoningEffort());
        assertFalse(request.logprobs());
        assertEquals(0, request.topLogprobs());
    }

    @Test
    public void overridesRoundTripAndTopLogprobsImpliesLogprobs() {
        GenerationRequest request =
                GenerationRequest.builder()
                        .prompt("x")
                        .thinkingMode(ThinkingMode.DISABLED)
                        .reasoningEffort(ReasoningEffort.DEFAULT)
                        .topLogprobs(5)
                        .build();
        assertEquals(ThinkingMode.DISABLED, request.thinkingMode());
        assertEquals(ReasoningEffort.DEFAULT, request.reasoningEffort());
        assertTrue("asking for alternatives asks for logprobs", request.logprobs());
        assertEquals(5, request.topLogprobs());
    }

    @Test
    public void topLogprobsOutsideTheOpenAiCapIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").topLogprobs(-1).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").topLogprobs(21).build());
    }

    @Test
    public void encoderOverridesResolvePerRequestAndLeaveTheSessionEncoderAlone() {
        ConversationEncoder session =
                new ConversationEncoder(null, ThinkingMode.ENABLED, ReasoningEffort.XHIGH);
        assertSame("no overrides, the session encoder itself", session,
                session.withOverrides(null, null));
        assertSame("repeating the session values is also no override", session,
                session.withOverrides(ThinkingMode.ENABLED, ReasoningEffort.XHIGH));

        ConversationEncoder off = session.withOverrides(ThinkingMode.DISABLED, null);
        assertEquals(ThinkingMode.DISABLED, off.thinkingMode());
        assertEquals("an unset effort inherits", ReasoningEffort.XHIGH, off.reasoningEffort());
        assertEquals(
                "the session encoder keeps its own mode for the next request",
                ThinkingMode.ENABLED,
                session.thinkingMode());

        ConversationEncoder low = session.withOverrides(null, ReasoningEffort.LOW);
        assertEquals(ThinkingMode.ENABLED, low.thinkingMode());
        assertEquals(ReasoningEffort.LOW, low.reasoningEffort());
    }

    /** A tokenizer whose tokens are their own one-character texts; token 9 is a stop token. */
    private static Tokenizer stubTokenizer() {
        return new Tokenizer() {
            @Override public String regexPattern() { return ""; }
            @Override public Map<String, Integer> getSpecialTokens() { return Map.of(); }
            @Override public boolean isSpecialToken(int token) { return token == 9; }
            @Override public boolean shouldDisplayToken(int token) { return token != 9; }
            @Override public List<Integer> encode(String text, Set<String> allowed) { return List.of(); }
            @Override public List<Integer> encodeAsList(String text) { return List.of(); }
            @Override public String decode(List<Integer> tokens) {
                StringBuilder out = new StringBuilder();
                for (int token : tokens) {
                    out.append((char) ('a' + token));
                }
                return out.toString();
            }
        };
    }

    @Test
    public void eventStreamPairsEachTokenWithItsCapturedEntryAndDropsTheStops() {
        List<SampledTokenLogprob> sampled =
                List.of(
                        new SampledTokenLogprob(0, -0.5, List.of(new SampledTokenLogprob.Top(1, -1.5))),
                        new SampledTokenLogprob(1, -2.0, List.of()),
                        new SampledTokenLogprob(9, -0.1, List.of()));
        int[] next = {0};
        List<GenerationEvent> events = new java.util.ArrayList<>();
        TokenEventStream stream =
                new TokenEventStream(
                        stubTokenizer(),
                        Set.of(9),
                        events::add,
                        null,
                        token -> sampled.get(next[0]++));
        stream.accept(0);
        stream.accept(1);
        stream.accept(9);
        assertEquals("ab", stream.finish());

        assertEquals("the terminal stop token emits no event and no entry", 2, events.size());
        assertEquals(-0.5, events.get(0).tokenLogprob().logprob(), 0.0);
        assertEquals("a", events.get(0).tokenLogprob().text());
        assertEquals(
                "alternatives carry their own decoded text",
                "b",
                events.get(0).tokenLogprob().topLogprobs().get(0).text());
        assertEquals(-2.0, events.get(1).tokenLogprob().logprob(), 0.0);

        assertEquals(
                events.stream().map(GenerationEvent::tokenLogprob).toList(),
                DelegatingSession.tokenLogprobsOf(events, List.of(0, 1)));
    }

    @Test
    public void aMismatchedCaptureReportsNoEntriesRatherThanShiftingThem() {
        List<GenerationEvent> events =
                List.of(
                        new GenerationEvent(0, "a", new TokenLogprob(0, "a", -0.5, List.of())),
                        new GenerationEvent(1, "b", null));
        assertTrue(DelegatingSession.tokenLogprobsOf(events, List.of(0, 1)).isEmpty());
        assertTrue(
                DelegatingSession.tokenLogprobsOf(events, List.of(0, 1, 2)).isEmpty());
        assertTrue(DelegatingSession.tokenLogprobsOf(null, List.of()).isEmpty());
    }
}

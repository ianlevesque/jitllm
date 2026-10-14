package org.beehive.jitllm.server;

import static org.junit.Assert.*;

import java.util.Map;
import org.beehive.jitllm.api.ReasoningEffort;
import org.beehive.jitllm.api.ThinkingMode;
import org.junit.Test;

public final class OpenAIServerReasoningTest {

    // ── Non-streaming split ──────────────────────────────────────────────────

    @Test
    public void splitsLeadingThinkBlock() {
        var split = OpenAIServer.splitReasoning("<think>\nreasoning here\n</think>\n\nThe answer.");
        assertEquals("reasoning here", split.reasoningContent());
        assertEquals("The answer.", split.content());
    }

    @Test
    public void noThinkBlockIsByteForByteUnchanged() {
        String text = "  A plain answer.\nIt quotes <think> and </think> mid-text.\n";
        var split = OpenAIServer.splitReasoning(text);
        assertNull(split.reasoningContent());
        assertEquals(text, split.content());
    }

    @Test
    public void unclosedThinkAtTokenCapIsAllReasoning() {
        var split = OpenAIServer.splitReasoning("<think>\nstill thinking when cut off");
        assertEquals("still thinking when cut off", split.reasoningContent());
        assertEquals("", split.content());
    }

    @Test
    public void emptyThinkBlockYieldsEmptyReasoning() {
        var split = OpenAIServer.splitReasoning("<think>\n\n</think>\n\nanswer");
        assertEquals("", split.reasoningContent());
        assertEquals("answer", split.content());
    }

    @Test
    public void onlySeparatorNewlinesAreStripped() {
        var split = OpenAIServer.splitReasoning("<think> keep space </think> answer  \n");
        assertEquals(" keep space ", split.reasoningContent());
        assertEquals(" answer  \n", split.content());
    }

    @Test
    public void nullTextSplitsToEmptyContent() {
        var split = OpenAIServer.splitReasoning(null);
        assertNull(split.reasoningContent());
        assertEquals("", split.content());
    }

    // ── Streaming split ──────────────────────────────────────────────────────

    private static String[] collect(OpenAIServer.ReasoningStreamParser parser, String... pieces) {
        var reasoning = new StringBuilder();
        var content = new StringBuilder();
        for (String piece : pieces) {
            for (var delta : parser.feed(piece)) {
                (delta.reasoning() ? reasoning : content).append(delta.text());
            }
        }
        for (var delta : parser.finish()) {
            (delta.reasoning() ? reasoning : content).append(delta.text());
        }
        return new String[] {reasoning.toString(), content.toString()};
    }

    @Test
    public void streamingHandlesTagsSplitAcrossPieces() {
        var parser = new OpenAIServer.ReasoningStreamParser();
        // Reasoning must not surface as content before the closing tag is decidable.
        for (String piece : new String[] {"<th", "ink>\nrea", "son\n</th"}) {
            for (var delta : parser.feed(piece)) {
                assertTrue("content leaked before </think>: " + delta.text(), delta.reasoning());
            }
        }
        String[] all =
                collect(
                        new OpenAIServer.ReasoningStreamParser(),
                        "<th", "ink>\nrea", "son\n</th", "ink>\n\nans", "wer");
        assertEquals("reason", all[0]);
        assertEquals("answer", all[1]);
    }

    @Test
    public void streamingHandlesOneCharacterPieces() {
        String text = "<think>\nx\n</think>\n\ny";
        var pieces = new String[text.length()];
        for (int i = 0; i < text.length(); i++) {
            pieces[i] = text.substring(i, i + 1);
        }
        String[] all = collect(new OpenAIServer.ReasoningStreamParser(), pieces);
        assertEquals("x", all[0]);
        assertEquals("y", all[1]);
    }

    @Test
    public void streamingWithoutThinkBlockIsAllContent() {
        String[] all =
                collect(new OpenAIServer.ReasoningStreamParser(), "Hel", "lo <think> quoted", "!");
        assertEquals("", all[0]);
        assertEquals("Hello <think> quoted!", all[1]);
    }

    @Test
    public void streamingUnclosedThinkFlushesAsReasoning() {
        String[] all = collect(new OpenAIServer.ReasoningStreamParser(), "<think>", "\nabc", "def");
        assertEquals("abcdef", all[0]);
        assertEquals("", all[1]);
    }

    @Test
    public void streamingPartialTagAtEndIsContent() {
        String[] all = collect(new OpenAIServer.ReasoningStreamParser(), "<th");
        assertEquals("", all[0]);
        assertEquals("<th", all[1]);
    }

    @Test
    public void streamingMatchesNonStreamingSplit() {
        String text = "<think>\nline one\nline two\n</think>\n\nDone.\n";
        String[] all =
                collect(new OpenAIServer.ReasoningStreamParser(), text.substring(0, 11),
                        text.substring(11, 25), text.substring(25));
        var split = OpenAIServer.splitReasoning(text);
        assertEquals(split.reasoningContent(), all[0]);
        assertEquals(split.content(), all[1]);
    }
    // ── logprobs / top_logprobs / n ──────────────────────────────────────────

    @Test
    public void logprobFieldsParsePerEndpointSpelling() {
        var chat = OpenAIServer.parseLogprobRequest(true, Boolean.TRUE, 3, null, false, true);
        assertNull(chat.error());
        assertTrue(chat.enabled());
        assertEquals(3, chat.topLogprobs());
        assertEquals(1, chat.n());

        var completions = OpenAIServer.parseLogprobRequest(false, 5, null, 2, false, true);
        assertNull(completions.error());
        assertTrue("a completions count enables logprobs", completions.enabled());
        assertEquals(5, completions.topLogprobs());
        assertEquals(2, completions.n());

        var noop = OpenAIServer.parseLogprobRequest(true, Boolean.FALSE, 0, 1, false, true);
        assertNull(noop.error());
        assertFalse(noop.enabled());
    }

    @Test
    public void logprobFieldErrorsAreNamed() {
        assertNotNull(
                OpenAIServer.parseLogprobRequest(true, Boolean.FALSE, 3, null, false, true).error());
        assertNotNull(
                OpenAIServer.parseLogprobRequest(true, Boolean.TRUE, 21, null, false, true).error());
        assertNotNull(
                OpenAIServer.parseLogprobRequest(true, 5, null, null, false, true).error());
        assertNotNull(
                OpenAIServer.parseLogprobRequest(false, -1, null, null, false, true).error());
        assertNotNull(
                OpenAIServer.parseLogprobRequest(true, null, null, 0, false, true).error());
        assertTrue(
                OpenAIServer.parseLogprobRequest(true, Boolean.TRUE, null, 2, true, true)
                        .error()
                        .contains("'n' > 1"));
        assertTrue(
                "the engine path refuses instead of dropping",
                OpenAIServer.parseLogprobRequest(true, Boolean.TRUE, null, null, false, false)
                        .error()
                        .contains("continuous-batching"));
        assertNull(
                "a disabled request is servable anywhere",
                OpenAIServer.parseLogprobRequest(true, Boolean.FALSE, null, null, false, false)
                        .error());
    }

    // ── Per-request reasoning controls ───────────────────────────────────────

    @Test
    public void enableThinkingParsesToAPerRequestOverride() {
        var off = OpenAIServer.parseReasoningControls(
                Map.of("enable_thinking", false), null, true);
        assertNull(off.error());
        assertEquals(ThinkingMode.DISABLED, off.thinkingMode());
        assertNull(off.reasoningEffort());

        var on = OpenAIServer.parseReasoningControls(
                Map.of("enable_thinking", true), "low", true);
        assertNull(on.error());
        assertEquals(ThinkingMode.ENABLED, on.thinkingMode());
        assertEquals(ReasoningEffort.LOW, on.reasoningEffort());

        var unset = OpenAIServer.parseReasoningControls(null, null, true);
        assertNull(unset.error());
        assertFalse("no fields, no override: the session mode applies", unset.any());
    }

    @Test
    public void reasoningEffortSpellingsMapOntoTheFacadeEfforts() {
        assertEquals(
                ReasoningEffort.XHIGH,
                OpenAIServer.parseReasoningControls(null, "high", true).reasoningEffort());
        assertEquals(
                ReasoningEffort.XHIGH,
                OpenAIServer.parseReasoningControls(null, "xhigh", true).reasoningEffort());
        assertEquals(
                ReasoningEffort.MEDIUM,
                OpenAIServer.parseReasoningControls(null, "medium", true).reasoningEffort());
        assertEquals(
                ThinkingMode.DISABLED,
                OpenAIServer.parseReasoningControls(null, "none", true).thinkingMode());
        assertNotNull(OpenAIServer.parseReasoningControls(null, "extreme", true).error());
    }

    @Test
    public void contradictoryAndMalformedReasoningControlsAreRejected() {
        assertNotNull(
                OpenAIServer.parseReasoningControls(
                                Map.of("enable_thinking", false), "high", true)
                        .error());
        assertNotNull(
                OpenAIServer.parseReasoningControls(
                                Map.of("enable_thinking", true), "none", true)
                        .error());
        assertNotNull(OpenAIServer.parseReasoningControls("nope", null, true).error());
        assertNotNull(
                OpenAIServer.parseReasoningControls(
                                Map.of("enable_thinking", "yes"), null, true)
                        .error());
        assertNotNull(OpenAIServer.parseReasoningControls(null, 42, true).error());
        assertTrue(
                "the engine path refuses explicit controls",
                OpenAIServer.parseReasoningControls(
                                Map.of("enable_thinking", false), null, false)
                        .error()
                        .contains("continuous-batching"));
        assertNull(
                "but an absent control asks it for nothing",
                OpenAIServer.parseReasoningControls(null, null, false).error());
    }
}

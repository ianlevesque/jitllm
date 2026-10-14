package org.beehive.jitllm.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.junit.Test;

/**
 * The request checks that used to be missing. Each one here corresponds to a real failure the
 * server produced by accepting something it could not serve.
 */
public class OpenAIServerValidationTest {

    private static final String SERVED = "Qwen3.5-4B-MTP-Q4_0";
    private static final int CTX = 8192;

    private static String validate(String requested, int maxTokens, int promptChars) {
        return OpenAIServer.validationError(requested, SERVED, maxTokens, promptChars, CTX);
    }

    // ---- model name ----------------------------------------------------------------------

    @Test
    public void theServedModelIsAccepted() {
        assertNull(validate(SERVED, 256, 100));
    }

    /** OpenAI clients may omit it; the server has only one model, so absence is unambiguous. */
    @Test
    public void anAbsentModelIsAccepted() {
        assertNull(validate(null, 256, 100));
        assertNull(validate("", 256, 100));
        assertNull(validate("   ", 256, 100));
    }

    /**
     * The failure this replaces: a client asking for one model was answered by another, with
     * nothing in the response to say so.
     */
    @Test
    public void aDifferentModelIsRejectedAndNamesBoth() {
        String err = validate("ggml-org/Qwen3-1.7B-GGUF", 256, 100);
        assertTrue("must name the served model: " + err, err.contains(SERVED));
        assertTrue(
                "must name the requested model: " + err, err.contains("ggml-org/Qwen3-1.7B-GGUF"));
    }

    @Test
    public void modelMatchingIsExactNotCaseInsensitive() {
        assertTrue(validate(SERVED.toLowerCase(), 256, 100) != null);
    }

    // ---- context window ------------------------------------------------------------------

    @Test
    public void maxTokensFillingTheWholeWindowIsRejected() {
        String err = validate(SERVED, CTX, 10);
        assertTrue("must mention max_tokens: " + err, err.contains("max_tokens"));
        assertTrue("must mention the window: " + err, err.contains(String.valueOf(CTX)));
    }

    /**
     * The prompt that cannot fit even at a generous 8 characters per token: 104_000 / 8 = 13_000
     * minimum tokens against an 8k window.
     */
    @Test
    public void aPromptGrosslyOverTheWindowIsRejected() {
        String err = validate(SERVED, 256, 104_000);
        assertTrue("must report the estimate: " + err, err.contains("26000"));
        assertTrue("must report the characters: " + err, err.contains("104000"));
    }

    @Test
    public void aPromptThatFitsOnlyWithoutTheOutputBudgetIsRejected() {
        // 64_000 / 8 = 8000 minimum prompt tokens leaves 192 of the window, less than 256.
        String err = validate(SERVED, 256, 64_000);
        assertTrue("must explain the combination: " + err, err.contains("max_tokens"));
    }

    /**
     * Measured English averages ~6.5 characters per token, so a legitimate near-window prompt
     * overestimates badly at 4 chars/token: 1_692_159 characters is 261_872 real tokens and
     * must reach tokenization instead of being refused by the heuristic.
     */
    @Test
    public void aFullContextEnglishPromptIsAccepted() {
        assertNull(
                OpenAIServer.validationError(SERVED, SERVED, 256, 1_692_159, 262_144));
    }

    @Test
    public void aComfortablePromptIsAccepted() {
        assertNull(validate(SERVED, 256, 4_000));
    }

    /** With no known window every context check is skipped rather than guessed at. */
    @Test
    public void anUnknownContextSkipsTheContextChecks() {
        assertNull(
                OpenAIServer.validationError(
                        SERVED, SERVED, 99_999, 10_000_000, OpenAIServer.UNKNOWN_CONTEXT_LENGTH));
    }

    /** A wrong model is still rejected when the window is unknown. */
    @Test
    public void anUnknownContextStillValidatesTheModel() {
        assertTrue(
                OpenAIServer.validationError(
                                "other", SERVED, 8, 8, OpenAIServer.UNKNOWN_CONTEXT_LENGTH)
                        != null);
    }

    // ---- promptCharacters ----------------------------------------------------------------

    @Test
    public void promptCharactersSumsEveryMessage() {
        List<ChatMessage> messages =
                List.of(
                        ChatMessage.of(ChatRole.SYSTEM, "12345"),
                        ChatMessage.of(ChatRole.USER, "1234567890"));

        assertEquals(15, OpenAIServer.promptCharacters(messages));
    }

    @Test
    public void promptCharactersOfNoMessagesIsZero() {
        assertEquals(0, OpenAIServer.promptCharacters(List.of()));
    }

    @Test
    public void theEstimateRatioIsTheDocumentedOne() {
        assertEquals(4, OpenAIServer.ESTIMATED_CHARS_PER_TOKEN);
        assertEquals(8, OpenAIServer.MAX_CHARS_PER_TOKEN);
    }

    // ---- request logging -----------------------------------------------------------------

    @Test
    public void theRequestSummaryCarriesWhatIdentifiesTheCall() {
        String line =
                OpenAIServer.requestSummary(
                        "chatcmpl-7", "POST /v1/chat/completions", SERVED, 1234, 256, false);

        assertTrue(line.contains("chatcmpl-7"));
        assertTrue(line.contains("POST /v1/chat/completions"));
        assertTrue(line.contains(SERVED));
        assertTrue(line.contains("promptChars=1234"));
        assertTrue(line.contains("maxTokens=256"));
    }

    @Test
    public void aStreamingRequestIsMarkedAsSuch() {
        assertTrue(
                OpenAIServer.requestSummary("cmpl-1", "POST /v1/completions", SERVED, 10, 8, true)
                        .contains("stream"));
    }

    @Test
    public void aNonStreamingRequestIsNotMarkedStreaming() {
        assertTrue(
                !OpenAIServer.requestSummary("cmpl-1", "POST /v1/completions", SERVED, 10, 8, false)
                        .contains("stream"));
    }

    /** An omitted model is legal, so the line has to say so rather than print "null". */
    @Test
    public void anAbsentModelIsRenderedReadably() {
        String line =
                OpenAIServer.requestSummary("cmpl-1", "POST /v1/completions", null, 10, 8, false);

        assertTrue("should not print null: " + line, line.contains("(unset)"));
        assertTrue(!line.contains("null"));
    }
}

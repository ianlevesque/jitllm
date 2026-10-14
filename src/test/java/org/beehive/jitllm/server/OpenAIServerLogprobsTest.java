package org.beehive.jitllm.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.api.TokenLogprob;
import org.junit.Test;

/** The OpenAI {@code choices[].logprobs.content[]} wire shape built from facade entries. */
public final class OpenAIServerLogprobsTest {

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> content(Map<String, Object> payload) {
        return (List<Map<String, Object>>) payload.get("content");
    }

    @Test
    public void payloadCarriesTokenLogprobBytesAndAlternatives() {
        var payload =
                OpenAIServer.logprobsPayload(
                        List.of(
                                new TokenLogprob(
                                        17,
                                        "2",
                                        -0.25,
                                        List.of(
                                                new TokenLogprob.Alternative(17, "2", -0.25),
                                                new TokenLogprob.Alternative(18, "3", -1.5))),
                                new TokenLogprob(220, " ", -3.0, List.of())));
        var content = content(payload);
        assertEquals(2, content.size());

        Map<String, Object> first = content.get(0);
        assertEquals("2", first.get("token"));
        assertEquals(-0.25, (Double) first.get("logprob"), 0.0);
        assertEquals(List.of((int) '2'), first.get("bytes"));

        var tops = (List<Map<String, Object>>) first.get("top_logprobs");
        assertEquals(2, tops.size());
        assertEquals("3", tops.get(1).get("token"));
        assertEquals(-1.5, (Double) tops.get(1).get("logprob"), 0.0);
        assertEquals(List.of((int) '3'), tops.get(1).get("bytes"));

        assertEquals(List.of(32), content.get(1).get("bytes"));
        assertEquals(List.of(), content.get(1).get("top_logprobs"));
    }

    @Test
    public void bytesAreUtf8AndNullWhenTheTextCannotCarryThem() {
        var payload =
                OpenAIServer.logprobsPayload(
                        List.of(
                                new TokenLogprob(1, "é", -1.0, List.of()),
                                new TokenLogprob(2, "�", -2.0, List.of())));
        var content = content(payload);
        byte[] utf8 = "é".getBytes(StandardCharsets.UTF_8);
        assertEquals(List.of(utf8[0] & 0xFF, utf8[1] & 0xFF), content.get(0).get("bytes"));
        assertNull(
                "a replacement character's raw bytes are not recoverable",
                content.get(1).get("bytes"));
        assertTrue(content.get(1).containsKey("bytes"));
    }

    @Test
    public void anUnderflowedLogprobIsWrittenAsMinus9999NotInfinity() {
        var payload =
                OpenAIServer.logprobsPayload(
                        List.of(new TokenLogprob(1, "x", Double.NEGATIVE_INFINITY, List.of())));
        assertEquals(-9999.0, (Double) content(payload).get(0).get("logprob"), 0.0);
        assertTrue("the payload is valid JSON", !Json.write(payload).contains("Infinity"));
    }

    @Test
    public void anEmptyEntryListIsAnEmptyContentArray() {
        assertEquals(List.of(), content(OpenAIServer.logprobsPayload(List.of())));
    }
}

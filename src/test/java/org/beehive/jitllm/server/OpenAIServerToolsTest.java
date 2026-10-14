package org.beehive.jitllm.server;

import static org.junit.Assert.*;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.api.ChatContent;
import org.junit.Test;

public final class OpenAIServerToolsTest {
    @Test public void messageTextAcceptsStringsAndTextParts() {
        assertEquals("hi", OpenAIServer.messageText("hi"));
        assertEquals("", OpenAIServer.messageText(null));
        assertEquals("ab", OpenAIServer.messageText(List.of(
                Map.of("type", "text", "text", "a"), Map.of("type", "text", "text", "b"))));
    }
    @Test public void messageTextRejectsNonTextParts() {
        try {
            OpenAIServer.messageText(List.of(Map.of("type", "image_url", "image_url", Map.of("url", "x"))));
            fail("image part accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("image_url"));
        }
    }
    @Test public void parseToolsBuildsSpecsAndHonorsChoice() {
        var body = Map.<String, Object>of("tools", List.of(Map.of(
                "type", "function",
                "function", Map.of("name", "get_weather", "description", "Weather by city",
                        "parameters", Map.of("type", "object")))));
        var tools = OpenAIServer.parseTools(body);
        assertEquals(1, tools.size());
        assertEquals("get_weather", tools.getFirst().name());
        assertEquals("Weather by city", tools.getFirst().description());
        assertTrue(tools.getFirst().parametersJsonSchema().contains("\"object\""));
        assertTrue(OpenAIServer.parseTools(Map.of("tools", List.of(), "tool_choice", "none")).isEmpty());
        assertTrue(OpenAIServer.parseTools(Map.of()).isEmpty());
    }
    @Test public void parseToolsRejectsForcedChoiceAndNamelessTools() {
        try {
            OpenAIServer.parseTools(Map.of("tool_choice", Map.of("type", "function")));
            fail("forced tool_choice accepted");
        } catch (IllegalArgumentException expected) {}
        try {
            OpenAIServer.parseTools(Map.<String, Object>of("tools",
                    List.of(Map.of("type", "function", "function", Map.of("description", "x")))));
            fail("nameless tool accepted");
        } catch (IllegalArgumentException expected) {}
    }
    @Test public void contentPiecesDecodeDataUrlImages() {
        var pieces = OpenAIServer.contentPieces(List.of(
                Map.of("type", "text", "text", "look:"),
                Map.of("type", "image_url", "image_url", Map.of("url",
                        "data:image/png;base64," + java.util.Base64.getEncoder()
                                .encodeToString(new byte[]{(byte) 0x89, 'P', 'N', 'G'})))));
        assertEquals(2, pieces.size());
        assertTrue(pieces.get(0) instanceof ChatContent.Text);
        assertTrue(pieces.get(1) instanceof ChatContent.Image);
        assertArrayEquals(new byte[]{(byte) 0x89, 'P', 'N', 'G'},
                ((ChatContent.Image) pieces.get(1)).encodedBytes());
        assertEquals("look:", OpenAIServer.textOf(pieces));
    }
    @Test public void contentPiecesRejectBadImageUrls() {
        for (String url : new String[]{"file:///etc/passwd", "data:text/plain,hello", ""}) {
            try {
                OpenAIServer.contentPieces(List.of(Map.of(
                        "type", "image_url", "image_url", Map.of("url", url))));
                fail("bad image url accepted: " + url);
            } catch (IllegalArgumentException expected) {}
        }
    }
    @Test public void toolCallJsonHasOpenAiShape() {
        var call = new ChatContent.ToolCall("call_1", "get_weather", "{\"city\":\"Oslo\"}");
        var json = OpenAIServer.toolCallJson(call, null);
        assertEquals("call_1", json.get("id"));
        assertEquals("function", json.get("type"));
        assertEquals(Map.of("name", "get_weather", "arguments", "{\"city\":\"Oslo\"}"), json.get("function"));
        assertEquals(2, OpenAIServer.toolCallJson(call, 2).get("index"));
    }
}

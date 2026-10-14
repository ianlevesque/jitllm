package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Qwen4ChatFormat;
import org.beehive.jitllm.model.format.ToolCallExtract;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Flash-Next conversations through the facade's encoder, against the checkpoint's own chat
 * template ({@code nvidia/Qwen3.8-Flash-Next-NVFP4} snapshot {@code fc694b54}, {@code
 * chat_template.jinja}, identical to the {@code chat_template} string in its {@code
 * tokenizer_config.json}).
 *
 * <p>Each {@code EXPECTED_*} constant is that template rendered by Jinja2 under the Hugging Face
 * settings {@code render_expected.py} uses ({@code trim_blocks}/{@code lstrip_blocks}, {@code
 * tojson} as {@code json.dumps(..., ensure_ascii=False)}, {@code add_generation_prompt=True});
 * the same inputs through transformers' {@code apply_chat_template} render identically. The
 * format is exercised with a synthetic {@code tokenizer.json}: one token per GPT-2 byte, no
 * merges and the ChatML/tool/think added tokens, so encoded conversations decode back to exactly
 * the text that was rendered.
 *
 * <p>Two stated deviations of the Qwen 3.5 base format are normalised where they occur, exactly
 * as {@link ReasoningEffortTemplateTest} normalises them: with thinking on, the template's
 * generation prompt opens the thinking block ({@code <think>\n}), which the engine lets the model
 * write itself; and the template writes a {@code <think>} block (holding {@code
 * reasoning_content}, which no {@link ChatContent} piece can carry) in front of every replayed
 * assistant turn, which the engine does not. Every other byte must match.
 */
public class Qwen4ChatFormatTemplateTest {

    private Path directory;
    private Qwen4Tokenizer tokenizer;
    private Qwen4ChatFormat format;

    @Before
    public void setUp() throws IOException {
        directory = Files.createTempDirectory("qwen4-chat-template-test-");
        Files.writeString(directory.resolve("tokenizer.json"), syntheticTokenizerJson());
        tokenizer = Qwen4Tokenizer.load(directory);
        format = new Qwen4ChatFormat(tokenizer);
    }

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(directory.resolve("tokenizer.json"));
        Files.deleteIfExists(directory);
    }

    // ---- checkpoint-template cases, byte for byte ---------------------------------------------

    /** Thinking disabled is byte-identical with no normalisation at all. */
    @Test
    public void plainUserWithThinkingDisabledMatchesTheTemplateExactly() {
        assertEquals(
                EXPECTED_PLAIN_USER_THINKING_DISABLED,
                encode(
                        List.of(ChatMessage.of(ChatRole.USER, "What is the capital of France?")),
                        List.of(),
                        ThinkingMode.DISABLED,
                        ReasoningEffort.DEFAULT));
    }

    /** The template's default effort — and an explicit {@code xhigh} — is the xhigh instruction. */
    @Test
    public void plainUserByDefaultGetsTheXhighInstruction() {
        for (ReasoningEffort effort : List.of(ReasoningEffort.DEFAULT, ReasoningEffort.XHIGH)) {
            assertMatchesButForThinkBlocks(
                    EXPECTED_PLAIN_USER_DEFAULT,
                    encode(
                            List.of(
                                    ChatMessage.of(
                                            ChatRole.USER, "What is the capital of France?")),
                            List.of(),
                            ThinkingMode.DEFAULT,
                            effort));
        }
    }

    /** {@code medium} writes no instruction and no system turn; {@code low} writes its own. */
    @Test
    public void plainUserMediumAndLowEfforts() {
        assertMatchesButForThinkBlocks(
                EXPECTED_PLAIN_USER_MEDIUM,
                encode(
                        List.of(ChatMessage.of(ChatRole.USER, "What is the capital of France?")),
                        List.of(),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.MEDIUM));
        assertMatchesButForThinkBlocks(
                EXPECTED_PLAIN_USER_LOW,
                encode(
                        List.of(ChatMessage.of(ChatRole.USER, "What is the capital of France?")),
                        List.of(),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.LOW));
    }

    /** System, user, assistant, user: the instruction precedes the caller's system text. */
    @Test
    public void multiTurnMatchesTheTemplate() {
        assertMatchesButForThinkBlocks(
                EXPECTED_MULTITURN,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.SYSTEM, "You are a concise assistant."),
                                ChatMessage.of(ChatRole.USER, "What is 2 + 2?"),
                                ChatMessage.of(
                                        ChatRole.ASSISTANT, "2 + 2 = 4.\n\nAnything else?"),
                                ChatMessage.of(ChatRole.USER, "Summarise that in one word.")),
                        List.of(),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    /** The template trims rendered content ({@code | trim}); a padded user turn is trimmed too. */
    @Test
    public void whitespaceAroundUserContentIsTrimmedAsTheTemplateTrimsIt() {
        assertMatchesButForThinkBlocks(
                EXPECTED_WHITESPACE_USER,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.SYSTEM, "You are a concise assistant."),
                                ChatMessage.of(ChatRole.USER, "  Name one colour.\n")),
                        List.of(),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    /** A blank system turn still gets the instruction, with nothing after it. */
    @Test
    public void blankSystemMessageGetsOnlyTheReasoningInstruction() {
        assertMatchesButForThinkBlocks(
                EXPECTED_BLANK_SYSTEM,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.SYSTEM, ""),
                                ChatMessage.of(ChatRole.USER, "Hi")),
                        List.of(),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    /** Tool definitions, the instruction ahead of them and the system text after them. */
    @Test
    public void toolDefinitionsWithAndWithoutASystemMessage() {
        assertMatchesButForThinkBlocks(
                EXPECTED_TOOLS_SYSTEM_AND_USER,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.SYSTEM, "You are helpful."),
                                ChatMessage.of(ChatRole.USER, "What is the weather in Paris?")),
                        List.of(WEATHER),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
        assertMatchesButForThinkBlocks(
                EXPECTED_TOOLS_NO_SYSTEM_TWO_TOOLS,
                encode(
                        List.of(ChatMessage.of(ChatRole.USER, "Hi")),
                        List.of(WEATHER, CLOCK),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    /** A replayed XML tool call and its result, delivered as a user turn. */
    @Test
    public void toolCallAndResultMatchTheTemplate() {
        assertMatchesButForThinkBlocks(
                EXPECTED_CALL_AND_RESULT,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.USER, "What is the weather in Paris?"),
                                toolCallMessage("c1", "getWeather", "{\"city\": \"Paris\"}"),
                                toolResultMessage(
                                        "c1",
                                        "getWeather",
                                        "{\"temperature\":18,\"condition\":\"sunny\"}")),
                        List.of(WEATHER),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    /** Two calls in one turn, their two results in one user turn, an answer, a new question. */
    @Test
    public void twoToolCallsResultsAnswerAndUserMatchTheTemplate() {
        assertMatchesButForThinkBlocks(
                EXPECTED_TWO_CALLS_RESULTS_ANSWER_USER,
                encode(
                        List.of(
                                ChatMessage.of(ChatRole.USER, "Weather in Paris and the time?"),
                                new ChatMessage(
                                        ChatRole.ASSISTANT,
                                        List.of(
                                                new ChatContent.ToolCall(
                                                        "c1", "getWeather", "{\"city\": \"Paris\"}"),
                                                new ChatContent.ToolCall(
                                                        "c2", "get_current_time", "{}"))),
                                toolResultMessage("c1", "getWeather", "sunny"),
                                toolResultMessage("c2", "get_current_time", "12:00"),
                                ChatMessage.of(ChatRole.ASSISTANT, "It is sunny and noon."),
                                ChatMessage.of(ChatRole.USER, "Thanks")),
                        List.of(WEATHER, CLOCK),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT));
    }

    // ---- the format's own fragments, exactly as the template writes them ----------------------

    /** One message: ChatML header, trimmed content, {@code <|im_end|>} and a newline. */
    @Test
    public void aMessageIsItsTrimmedContentBetweenChatMlMarkers() {
        assertEquals(
                "<|im_start|>user\nName one colour.<|im_end|>\n",
                tokenizer.decode(
                        format.encodeMessage(
                                new ChatFormat.Message(
                                        ChatFormat.Role.USER, "  Name one colour.\n"))));
    }

    /** Consecutive results share one user turn; each result's content is trimmed. */
    @Test
    public void toolResultsShareOneUserTurnWithTrimmedContent() {
        assertEquals(
                "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n"
                        + "<tool_response>\n12:00\n</tool_response><|im_end|>\n",
                tokenizer.decode(
                        format.encodeToolResults(
                                List.of(
                                        new ChatFormat.ToolResult("c1", "getWeather", "  sunny\n"),
                                        new ChatFormat.ToolResult(
                                                "c2", "get_current_time", "12:00")))));
    }

    /**
     * Typed arguments as {@code <parameter>} bodies: strings verbatim, everything else as JSON.
     * The checkpoint render of this turn additionally carries its replayed-assistant {@code
     * <think>} block (and any assistant prose, which a {@link ChatContent.ToolCall} turn here
     * drops) before the call; neither is expressible through {@code encodeToolCallAssistantTurn},
     * so this fragment is the template's from {@code <tool_call>} on.
     */
    @Test
    public void typedToolCallArgumentsRenderAsTemplateParameters() {
        assertEquals(
                "<|im_start|>assistant\n<tool_call>\n<function=getWeather>\n"
                        + "<parameter=city>\nParis\n</parameter>\n"
                        + "<parameter=days>\n3\n</parameter>\n"
                        + "<parameter=exact>\ntrue\n</parameter>\n"
                        + "<parameter=bounds>\n{\"lo\": 1, \"hi\": 9}\n</parameter>\n"
                        + "</function>\n</tool_call><|im_end|>\n",
                tokenizer.decode(
                        format.encodeToolCallAssistantTurn(
                                List.of(
                                        new ToolCallExtract(
                                                "getWeather",
                                                "{\"city\": \"Paris\", \"days\": 3,"
                                                        + " \"exact\": true, \"bounds\":"
                                                        + " {\"lo\": 1, \"hi\": 9}}")))));
    }

    /** The reasoning-effort control: the template's strings, its default and its rejection. */
    @Test
    public void reasoningEffortInstructionsAreTheTemplatesVerbatim() {
        assertTrue(format.supportsToolCalling());
        assertTrue(format.supportsReasoningEffort());
        assertEquals("xhigh", format.defaultReasoningEffort());
        assertEquals(
                "Reasoning effort is set to xhigh. Please think carefully through the task,"
                        + " validate key assumptions, consider plausible alternatives, and"
                        + " prioritize correctness, consistency, and clarity in the final answer.",
                format.reasoningEffortInstructions("xhigh"));
        assertEquals("", format.reasoningEffortInstructions("medium"));
        assertEquals(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving"
                        + " directly to the conclusion without unnecessary elaboration.",
                format.reasoningEffortInstructions("low"));
        assertThrows(
                IllegalArgumentException.class,
                () -> format.reasoningEffortInstructions("bogus"));
    }

    // ---- image turns, as the template writes them and the processor expands them --------------

    /**
     * The template writes one {@code <|image_pad|>} placeholder per image; the image processor
     * expands it to the merged token count. Six here is the 96x64 grid (1,4,6) at merge 2.
     */
    @Test
    public void anImageTurnMatchesTheTemplateWithThePlaceholderExpanded() {
        assertEquals(
                expandImagePads(EXPECTED_IMAGE_TURN_FRAGMENT, 6),
                tokenizer.decode(
                        format.encodeRichMessage(
                                ChatFormat.Role.USER,
                                List.of(
                                        new ChatFormat.ContentPart.Image(6),
                                        new ChatFormat.ContentPart.Text("What is in this picture?")))));
    }

    /** Whitespace beside an image is content; only the turn's outer ends are trimmed. */
    @Test
    public void textAroundAnImageKeepsItsInnerWhitespaceAndTrimsTheEnds() {
        assertEquals(
                "<|im_start|>user\nLook: <|vision_start|>"
                        + "<|image_pad|>".repeat(2)
                        + "<|vision_end|> Describe it.<|im_end|>\n",
                tokenizer.decode(
                        format.encodeRichMessage(
                                ChatFormat.Role.USER,
                                List.of(
                                        new ChatFormat.ContentPart.Text("  Look: "),
                                        new ChatFormat.ContentPart.Image(2),
                                        new ChatFormat.ContentPart.Text(" Describe it.  ")))));
    }

    /** Two images keep their order and their own expansions. */
    @Test
    public void twoImagesExpandIndependentlyInOrder() {
        assertEquals(
                "<|im_start|>user\n<|vision_start|>"
                        + "<|image_pad|>".repeat(3)
                        + "<|vision_end|><|vision_start|>"
                        + "<|image_pad|>".repeat(5)
                        + "<|vision_end|>Compare them.<|im_end|>\n",
                tokenizer.decode(
                        format.encodeRichMessage(
                                ChatFormat.Role.USER,
                                List.of(
                                        new ChatFormat.ContentPart.Image(3),
                                        new ChatFormat.ContentPart.Image(5),
                                        new ChatFormat.ContentPart.Text("Compare them.")))));
    }

    /** A model with no tower refuses an image conversation instead of dropping the image. */
    @Test
    public void imagesAreRefusedWhenTheModelOwnsNoVisionTower() {
        ConversationEncoder encoder =
                new ConversationEncoder(
                        new Gemma4ToolConversationTest.FormatOnlyModel(format, false),
                        ThinkingMode.DEFAULT,
                        ReasoningEffort.DEFAULT);
        var withImage =
                List.of(
                        new ChatMessage(
                                ChatRole.USER, List.of(new ChatContent.Image(new byte[] {1, 2, 3}))));
        assertThrows(
                UnsupportedOperationException.class,
                () -> encoder.encodeConversation(withImage, List.of()));
        assertTrue(format.supportsImages());
    }

    /** The template's single placeholder, expanded to {@code counts} copies in order. */
    private static String expandImagePads(String rendered, int... counts) {
        StringBuilder expanded = new StringBuilder();
        int next = 0;
        for (String piece : rendered.split("<\\|image_pad\\|>", -1)) {
            if (next > 0) expanded.append("<|image_pad|>".repeat(counts[next - 1]));
            expanded.append(piece);
            next++;
        }
        assertEquals(counts.length, next - 1);
        return expanded.toString();
    }

    // ---- harness ------------------------------------------------------------------------------

    private String encode(
            List<ChatMessage> messages,
            List<ToolSpec> tools,
            ThinkingMode thinking,
            ReasoningEffort effort) {
        ConversationEncoder encoder =
                new ConversationEncoder(
                        new Gemma4ToolConversationTest.FormatOnlyModel(format, false),
                        thinking,
                        effort);
        return tokenizer.decode(encoder.encode(messages, tools));
    }

    /**
     * Equality after removing only the two stated think-block deviations from the template's
     * rendering: empty replayed-assistant blocks, and the trailing thinking opener of its
     * generation prompt.
     */
    private static void assertMatchesButForThinkBlocks(String expected, String actual) {
        String normalised = expected.replaceAll("<think>\n\n</think>\n\n(?=.)", "");
        assertTrue("the template opens the thinking block", normalised.endsWith("<think>\n"));
        assertEquals(normalised.substring(0, normalised.length() - "<think>\n".length()), actual);
    }

    private static ChatMessage toolCallMessage(String id, String name, String argumentsJson) {
        return new ChatMessage(
                ChatRole.ASSISTANT, List.of(new ChatContent.ToolCall(id, name, argumentsJson)));
    }

    private static ChatMessage toolResultMessage(String id, String name, String result) {
        return new ChatMessage(
                ChatRole.TOOL, List.of(new ChatContent.ToolResult(id, name, result)));
    }

    private static final ToolSpec WEATHER =
            new ToolSpec(
                    "getWeather",
                    "Returns the current weather for a city.",
                    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\","
                            + "\"description\":\"The city name\"}},\"required\":[\"city\"]}");

    private static final ToolSpec CLOCK =
            new ToolSpec(
                    "get_current_time",
                    "Returns the current time.",
                    "{\"type\":\"object\",\"properties\":{},\"required\":[]}");

    /**
     * A minimal HF {@code tokenizer.json} for {@link Qwen4Tokenizer#load}: the 256 GPT-2 byte
     * tokens, no merges, and the added tokens in the order the Qwen tokenizer expects its special
     * block to start ({@code <|endoftext|>} first). The split regex takes the whole text as one
     * chunk, which changes nothing at one token per byte.
     */
    static String syntheticTokenizerJson() {
        List<Integer> printable = new ArrayList<>();
        for (int c = '!'; c <= '~'; c++) {
            printable.add(c);
        }
        for (int c = '¡'; c <= '¬'; c++) {
            printable.add(c);
        }
        for (int c = '®'; c <= 'ÿ'; c++) {
            printable.add(c);
        }
        int[] byteChar = new int[256];
        for (int c : printable) {
            byteChar[c] = c;
        }
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            if (!printable.contains(b)) {
                byteChar[b] = 256 + extra++;
            }
        }
        StringBuilder json = new StringBuilder("{\"model\":{\"type\":\"BPE\",");
        json.append("\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",");
        json.append("\"vocab\":{");
        for (int b = 0; b < 256; b++) {
            if (b > 0) {
                json.append(',');
            }
            json.append(quotedChar(byteChar[b])).append(':').append(b);
        }
        json.append("},\"merges\":[]},\"added_tokens\":[");
        List<String> added =
                List.of(
                        "<|endoftext|>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "<tool_call>",
                        "</tool_call>",
                        "<tool_response>",
                        "</tool_response>",
                        "<think>",
                        "</think>",
                        "<|vision_start|>",
                        "<|image_pad|>",
                        "<|vision_end|>");
        for (int i = 0; i < added.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append("{\"id\":")
                    .append(256 + i)
                    .append(",\"content\":")
                    .append(quoted(added.get(i)))
                    .append('}');
        }
        json.append("],\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[");
        json.append(
                "{\"type\":\"Split\",\"behavior\":\"Isolated\","
                        + "\"pattern\":{\"Regex\":\"[\\\\s\\\\S]+\"}},");
        json.append("{\"type\":\"ByteLevel\"}]}}");
        return json.toString();
    }

    /** A one-character JSON string literal, every non-plain-ASCII character escaped. */
    private static String quotedChar(int codePoint) {
        return quoted(Character.toString(codePoint));
    }

    private static String quoted(String text) {
        StringBuilder out = new StringBuilder("\"");
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '"') {
                out.append("\\\"");
            } else if (c == '\\') {
                out.append("\\\\");
            } else if (c < 0x20 || c > 0x7e) {
                out.append(String.format("\\u%04x", (int) c));
            } else {
                out.append(c);
            }
        }
        return out.append('"').toString();
    }

    // ---- the checkpoint template's renderings, verbatim ----------------------------------------

    private static final String EXPECTED_PLAIN_USER_DEFAULT =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im"
                    + "_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_PLAIN_USER_THINKING_DISABLED =
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>as"
                    + "sistant\n<think>\n\n</think>\n\n";

    private static final String EXPECTED_PLAIN_USER_MEDIUM =
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>as"
                    + "sistant\n<think>\n";

    private static final String EXPECTED_PLAIN_USER_LOW =
            "<|im_start|>system\nReasoning effort is set to low. Keep your thinking br"
                    + "ief and focused, moving directly to the conclusion without unnecessary e"
                    + "laboration.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|i"
                    + "m_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_MULTITURN =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nWhat"
                    + " is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n2 + 2 = 4."
                    + "\n\nAnything else?<|im_end|>\n<|im_start|>user\nSummarise that in one word.<"
                    + "|im_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_WHITESPACE_USER =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nName"
                    + " one colour.<|im_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_BLANK_SYSTEM =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant"
                    + "\n<think>\n";

    private static final String EXPECTED_TOOLS_SYSTEM_AND_USER =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools"
                    + ">\n{\"type\": \"function\", \"function\": {\"name\": \"getWeather\", \"description\":"
                    + " \"Returns the current weather for a city.\", \"parameters\": {\"type\": \"obje"
                    + "ct\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city "
                    + "name\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a funct"
                    + "ion ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<fun"
                    + "ction=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</p"
                    + "arameter>\n<parameter=example_parameter_2>\nThis is the value for the seco"
                    + "nd parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</too"
                    + "l_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specifie"
                    + "d format: an inner <function=...></function> block must be nested within"
                    + " <tool_call></tool_call> XML tags\n- Required parameters MUST be specifie"
                    + "d\n- You may provide optional reasoning for your function call in natural"
                    + " language BEFORE the function call, but NOT after\n- If there is no funct"
                    + "ion call available, answer the question like normal with your current kn"
                    + "owledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou "
                    + "are helpful.<|im_end|>\n<|im_start|>user\nWhat is the weather in Paris?<|i"
                    + "m_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_TOOLS_NO_SYSTEM_TWO_TOOLS =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools"
                    + ">\n{\"type\": \"function\", \"function\": {\"name\": \"getWeather\", \"description\":"
                    + " \"Returns the current weather for a city.\", \"parameters\": {\"type\": \"obje"
                    + "ct\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city "
                    + "name\"}}, \"required\": [\"city\"]}}}\n{\"type\": \"function\", \"function\": {\"name\""
                    + ": \"get_current_time\", \"description\": \"Returns the current time.\", \"para"
                    + "meters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}\n</tools>"
                    + "\n\nIf you choose to call a function ONLY reply in the following format wi"
                    + "th NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=e"
                    + "xample_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>"
                    + "\nThis is the value for the second parameter\nthat can span\nmultiple lines"
                    + "\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function"
                    + " calls MUST follow the specified format: an inner <function=...></functi"
                    + "on> block must be nested within <tool_call></tool_call> XML tags\n- Requi"
                    + "red parameters MUST be specified\n- You may provide optional reasoning fo"
                    + "r your function call in natural language BEFORE the function call, but N"
                    + "OT after\n- If there is no function call available, answer the question l"
                    + "ike normal with your current knowledge and do not tell the user about fu"
                    + "nction calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_s"
                    + "tart|>assistant\n<think>\n";

    private static final String EXPECTED_CALL_AND_RESULT =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools"
                    + ">\n{\"type\": \"function\", \"function\": {\"name\": \"getWeather\", \"description\":"
                    + " \"Returns the current weather for a city.\", \"parameters\": {\"type\": \"obje"
                    + "ct\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city "
                    + "name\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a funct"
                    + "ion ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<fun"
                    + "ction=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</p"
                    + "arameter>\n<parameter=example_parameter_2>\nThis is the value for the seco"
                    + "nd parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</too"
                    + "l_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specifie"
                    + "d format: an inner <function=...></function> block must be nested within"
                    + " <tool_call></tool_call> XML tags\n- Required parameters MUST be specifie"
                    + "d\n- You may provide optional reasoning for your function call in natural"
                    + " language BEFORE the function call, but NOT after\n- If there is no funct"
                    + "ion call available, answer the question like normal with your current kn"
                    + "owledge and do not tell the user about function calls\n</IMPORTANT><|im_e"
                    + "nd|>\n<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start"
                    + "|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=getWeather>\n<parame"
                    + "ter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_sta"
                    + "rt|>user\n<tool_response>\n{\"temperature\":18,\"condition\":\"sunny\"}\n</tool_r"
                    + "esponse><|im_end|>\n<|im_start|>assistant\n<think>\n";

    private static final String EXPECTED_TWO_CALLS_RESULTS_ANSWER_USER =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefu"
                    + "lly through the task, validate key assumptions, consider plausible alter"
                    + "natives, and prioritize correctness, consistency, and clarity in the fin"
                    + "al answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools"
                    + ">\n{\"type\": \"function\", \"function\": {\"name\": \"getWeather\", \"description\":"
                    + " \"Returns the current weather for a city.\", \"parameters\": {\"type\": \"obje"
                    + "ct\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city "
                    + "name\"}}, \"required\": [\"city\"]}}}\n{\"type\": \"function\", \"function\": {\"name\""
                    + ": \"get_current_time\", \"description\": \"Returns the current time.\", \"para"
                    + "meters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}\n</tools>"
                    + "\n\nIf you choose to call a function ONLY reply in the following format wi"
                    + "th NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=e"
                    + "xample_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>"
                    + "\nThis is the value for the second parameter\nthat can span\nmultiple lines"
                    + "\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function"
                    + " calls MUST follow the specified format: an inner <function=...></functi"
                    + "on> block must be nested within <tool_call></tool_call> XML tags\n- Requi"
                    + "red parameters MUST be specified\n- You may provide optional reasoning fo"
                    + "r your function call in natural language BEFORE the function call, but N"
                    + "OT after\n- If there is no function call available, answer the question l"
                    + "ike normal with your current knowledge and do not tell the user about fu"
                    + "nction calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWeather in Paris an"
                    + "d the time?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_cal"
                    + "l>\n<function=getWeather>\n<parameter=city>\nParis\n</parameter>\n</function>"
                    + "\n</tool_call>\n<tool_call>\n<function=get_current_time>\n</function>\n</tool"
                    + "_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny\n</tool_response>"
                    + "\n<tool_response>\n12:00\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                    + "<think>\n\n</think>\n\nIt is sunny and noon.<|im_end|>\n<|im_start|>user\nThan"
                    + "ks<|im_end|>\n<|im_start|>assistant\n<think>\n";

    /**
     * One user turn of the template's rendering for {@code [{image}, {text "What is in this
     * picture?"}]} (Jinja2, {@code enable_thinking=False}); the single {@code <|image_pad|>}
     * is the placeholder the processor expands.
     */
    private static final String EXPECTED_IMAGE_TURN_FRAGMENT =
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    + "What is in this picture?<|im_end|>\n";
}



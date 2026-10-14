package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.beehive.jitllm.inference.state.MultimodalPromptPlan;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.VisionModel;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.ToolCallExtract;
import org.beehive.jitllm.model.vision.FlashNextVisionConfig;
import org.beehive.jitllm.model.vision.PreprocessedImage;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * Turns a facade conversation into the tokens the model sees.
 *
 * <p>Package-private, and the <b>only</b> place the facade names {@link ChatFormat}. Formatting
 * stays model-owned and internal [§9]: no facade type carries a chat format, a template or a
 * stop-token set, and a caller cannot influence how a family renders a turn.
 *
 * <p>The assembly order is the one both integrations already use — tool definitions into the system
 * message, or into the first user message for formats that want them there; then each turn; then an
 * assistant header to prime the response. It is copied deliberately rather than reinvented: it is
 * the shape that has been producing correct tool calls in their tests, and a second opinion about
 * where tool JSON belongs would be a new defect rather than a new idea.
 */
final class ConversationEncoder {

    private final Model model;
    private final ThinkingMode thinkingMode;
    private final ReasoningEffort reasoningEffort;

    ConversationEncoder(Model model, ThinkingMode thinkingMode) {
        this(model, thinkingMode, ReasoningEffort.DEFAULT);
    }

    ConversationEncoder(Model model, ThinkingMode thinkingMode, ReasoningEffort reasoningEffort) {
        this.model = model;
        this.thinkingMode = thinkingMode;
        this.reasoningEffort = reasoningEffort;
    }

    ThinkingMode thinkingMode() {
        return thinkingMode;
    }

    ReasoningEffort reasoningEffort() {
        return reasoningEffort;
    }

    /**
     * The encoder for one request whose reasoning controls override this session's, or {@code
     * this} when neither override is set (or both repeat this encoder's values). An encoder is
     * only a model reference and the two resolved values — building one per request rebuilds
     * nothing in the model, its state or its plans; it only changes what {@link
     * #reasoningInstructions()} and {@link #appendThinkingControl(List)} render for that
     * request's prompt. A {@code null} argument inherits this encoder's value.
     */
    ConversationEncoder withOverrides(ThinkingMode thinkingOverride, ReasoningEffort effortOverride) {
        ThinkingMode thinking = thinkingOverride == null ? thinkingMode : thinkingOverride;
        ReasoningEffort effort = effortOverride == null ? reasoningEffort : effortOverride;
        return thinking == thinkingMode && effort == reasoningEffort
                ? this
                : new ConversationEncoder(model, thinking, effort);
    }

    /**
     * The reasoning-effort instructions this conversation's system turn opens with, or an empty
     * string.
     *
     * <p>Only for a family whose template has the control, and only while thinking is not disabled:
     * the template writes them when {@code enable_thinking} is undefined or true. {@link
     * ReasoningEffort#DEFAULT} is the template's own default, so a caller who says nothing gets
     * what llama.cpp and Ollama render.
     */
    String reasoningInstructions() {
        ChatFormat chatFormat = model.chatFormat();
        if (!chatFormat.supportsReasoningEffort() || thinkingMode == ThinkingMode.DISABLED) {
            return "";
        }
        String effort =
                reasoningEffort.isExplicit()
                        ? reasoningEffort.name().toLowerCase(java.util.Locale.ROOT)
                        : chatFormat.defaultReasoningEffort();
        return chatFormat.reasoningEffortInstructions(effort);
    }

    /**
     * A conversation's tokens plus, when it carries images, the plan that splices their tower
     * rows and MRoPE positions into those tokens. {@code imagePlan} is null exactly when no
     * message contains an image, which is what lets the session keep its text-only prefix
     * reuse: token equality cannot distinguish two different images of the same shape, so an
     * image-bearing encoding is never partially reused.
     */
    record EncodedConversation(List<Integer> tokens, MultimodalPromptPlan imagePlan) {
        boolean hasImages() {
            return imagePlan != null;
        }
    }

    /** One image after its single preprocess + tower pass, awaiting its span in the tokens. */
    private record EncodedImage(
            org.beehive.jitllm.model.vision.ImageGrid grid, float[][] embeddings) {}

    /**
     * The complete model input for a conversation — every token, in order.
     *
     * <p>This whole list is what prefix reuse compares [A1], which is why the tool specifications
     * are encoded here rather than being handled elsewhere: attaching a tool changes the system
     * content, so it changes the encoded input from its first tokens.
     *
     * <p>Image-bearing conversations are refused here: their tokens are only half the input
     * (the tower rows are the other half), and a caller that fed the tokens alone would run
     * image placeholders through the text embedding table. {@link
     * #encodeConversation(List, List)} returns both halves.
     */
    List<Integer> encode(List<ChatMessage> messages, List<ToolSpec> tools) {
        EncodedConversation encoded = encodeConversation(messages, tools);
        if (encoded.hasImages()) {
            throw new UnsupportedOperationException(
                    "image content needs the staged encoding path (encodeConversation);"
                            + " its tokens alone are not a runnable input");
        }
        return encoded.tokens();
    }

    /** As {@link #encode(List, List)}, with the image splice plan when the conversation has images. */
    EncodedConversation encodeConversation(List<ChatMessage> messages, List<ToolSpec> tools) {
        ChatFormat chatFormat = model.chatFormat();
        String toolsJson = tools.isEmpty() ? null : buildToolsJson(tools);
        if (toolsJson != null && !chatFormat.supportsToolCalling()) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.TOOLS_UNSUPPORTED.message("tool calling is not supported by ")
                            + model.getModelType()
                            + "'s chat format ("
                            + chatFormat.getClass().getSimpleName()
                            + ")");
        }

        // Images are checked the same way tools are, up front: a family without a vision path
        // (or a model that owns no tower) refuses the request rather than encoding around the
        // pieces it cannot run. ChatMessage already confines images to USER turns.
        boolean anyImages =
                messages.stream()
                        .flatMap(message -> message.content().stream())
                        .anyMatch(ChatContent.Image.class::isInstance);
        VisionModel visionModel = null;
        FlashNextVisionConfig visionConfig = null;
        if (anyImages) {
            if (!chatFormat.supportsImages() || !(model instanceof VisionModel vision)) {
                throw new UnsupportedOperationException(
                        "image content is not supported by "
                                + model.getModelType()
                                + "'s chat format ("
                                + chatFormat.getClass().getSimpleName()
                                + ")");
            }
            visionModel = vision;
            try {
                visionConfig = visionModel.visionConfig();
            } catch (java.io.IOException failure) {
                throw new java.io.UncheckedIOException(failure);
            }
        }
        List<EncodedImage> encodedImages = new ArrayList<>();

        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(chatFormat.getBeginOfText());
        }

        boolean injectInUserMessage = toolsJson != null && chatFormat.injectsToolsInUserMessage();
        boolean hasSystemMessage = messages.stream().anyMatch(m -> m.role() == ChatRole.SYSTEM);
        boolean toolsInjected = false;
        // Written once, at the front of the first system turn, as the template writes it. Formats
        // without the control never produce any (reasoningInstructions() is empty for them).
        String reasoningInstructions = reasoningInstructions();

        // A format that wants tools in the system message, and a conversation with no system turn
        // to put them in, gets one. So does a format that puts them in the user message but still
        // writes a system turn for them (Llama's environment lines).
        if (toolsJson != null
                && !hasSystemMessage
                && model.shouldAddSystemPrompt()
                && (!injectInUserMessage || !chatFormat.toolSystemMessagePrefix().isEmpty())) {
            tokens.addAll(
                    chatFormat.encodeToolSystemMessage(null, toolsJson, reasoningInstructions));
            reasoningInstructions = "";
            toolsInjected = !injectInUserMessage;
        }

        // A conversation with no system turn still gets one for the instructions, which is what
        // the template does when it has instructions and no system message.
        if (!reasoningInstructions.isEmpty() && !hasSystemMessage) {
            tokens.addAll(
                    chatFormat.encodeMessage(
                            new ChatFormat.Message(ChatFormat.Role.SYSTEM, reasoningInstructions)));
            reasoningInstructions = "";
        }

        // Whether an assistant turn is open: a format that writes tool results inside the turn
        // that made the calls leaves it open for the answer (Gemma 4). Never set otherwise.
        boolean openAssistantTurn = false;

        for (int index = 0; index < messages.size(); index++) {
            ChatMessage message = messages.get(index);
            if (openAssistantTurn
                    && message.role() != ChatRole.ASSISTANT
                    && message.role() != ChatRole.TOOL) {
                tokens.addAll(chatFormat.encodeOpenAssistantTurnEnd());
                openAssistantTurn = false;
            }
            switch (message.role()) {
                case USER -> {
                    if (message.content().stream().anyMatch(ChatContent.Image.class::isInstance)) {
                        String toolPrefix =
                                injectInUserMessage && !toolsInjected
                                        ? chatFormat.toolFirstUserMessagePrefix(toolsJson)
                                        : null;
                        tokens.addAll(
                                encodeUserWithImages(
                                        message, toolPrefix, visionModel, visionConfig, encodedImages));
                        toolsInjected |= toolPrefix != null;
                        continue;
                    }
                    String content = textOf(message);
                    if (injectInUserMessage && !toolsInjected) {
                        content = chatFormat.toolFirstUserMessagePrefix(toolsJson) + content;
                        toolsInjected = true;
                    }
                    tokens.addAll(
                            chatFormat.encodeMessage(
                                    new ChatFormat.Message(ChatFormat.Role.USER, content)));
                }
                case SYSTEM -> {
                    if (!model.shouldAddSystemPrompt()) {
                        continue; // this family has no system turn; dropping it is its rule
                    }
                    String content = textOf(message);
                    if (toolsJson != null) {
                        tokens.addAll(
                                chatFormat.encodeToolSystemMessage(
                                        content, toolsJson, reasoningInstructions));
                        reasoningInstructions = "";
                        toolsInjected |= !injectInUserMessage;
                        continue;
                    }
                    if (!reasoningInstructions.isEmpty()) {
                        content = reasoningInstructions + "\n\n" + content.strip();
                        reasoningInstructions = "";
                    }
                    tokens.addAll(
                            chatFormat.encodeMessage(
                                    new ChatFormat.Message(ChatFormat.Role.SYSTEM, content)));
                }
                case ASSISTANT -> {
                    List<ToolCallExtract> calls =
                            message.content().stream()
                                    .filter(ChatContent.ToolCall.class::isInstance)
                                    .map(ChatContent.ToolCall.class::cast)
                                    .map(
                                            call ->
                                                    new ToolCallExtract(
                                                            call.name(),
                                                            call.argumentsJson(),
                                                            Optional.of(call.id())))
                                    .toList();
                    if (!calls.isEmpty()) {
                        tokens.addAll(
                                openAssistantTurn
                                        ? chatFormat.encodeToolCallContinuation(calls)
                                        : chatFormat.encodeToolCallAssistantTurn(calls));
                        openAssistantTurn = chatFormat.toolResultsStayInAssistantTurn();
                    } else if (openAssistantTurn) {
                        tokens.addAll(chatFormat.encodeAssistantContinuation(textOf(message)));
                        openAssistantTurn = false;
                    } else {
                        tokens.addAll(
                                chatFormat.encodeMessage(
                                        new ChatFormat.Message(
                                                ChatFormat.Role.ASSISTANT, textOf(message))));
                    }
                }
                case TOOL -> {
                    // The whole run of consecutive results at once: some templates wrap a run in
                    // a single turn.
                    List<ChatFormat.ToolResult> results = new ArrayList<>();
                    while (true) {
                        for (ChatContent piece : messages.get(index).content()) {
                            ChatContent.ToolResult result = (ChatContent.ToolResult) piece;
                            results.add(
                                    new ChatFormat.ToolResult(
                                            result.id(), result.name(), result.resultJson()));
                        }
                        if (index + 1 < messages.size()
                                && messages.get(index + 1).role() == ChatRole.TOOL) {
                            index++;
                        } else {
                            break;
                        }
                    }
                    tokens.addAll(chatFormat.encodeToolResults(results));
                }
            }
        }

        // Prime the model to start an assistant turn — unless one is still open, in which case the
        // model's answer continues it, as the template's generation prompt does after tool results.
        if (!openAssistantTurn) {
            tokens.addAll(
                    chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            appendThinkingControl(tokens);
        }
        if (encodedImages.isEmpty()) {
            return new EncodedConversation(tokens, null);
        }
        return new EncodedConversation(
                tokens, imagePlan(tokens, encodedImages, visionConfig));
    }

    /**
     * One image-bearing user turn: every image is preprocessed and run through the tower
     * exactly once here, in content order; the format only ever sees the merged token count
     * the grid implies. The embeddings wait in {@code encodedImages} until the finished token
     * list fixes their spans.
     */
    private List<Integer> encodeUserWithImages(
            ChatMessage message,
            String toolPrefix,
            VisionModel visionModel,
            FlashNextVisionConfig visionConfig,
            List<EncodedImage> encodedImages) {
        List<ChatFormat.ContentPart> parts = new ArrayList<>();
        if (toolPrefix != null && !toolPrefix.isEmpty()) {
            parts.add(new ChatFormat.ContentPart.Text(toolPrefix));
        }
        for (ChatContent piece : message.content()) {
            switch (piece) {
                case ChatContent.Text text -> parts.add(new ChatFormat.ContentPart.Text(text.text()));
                case ChatContent.Image image -> {
                    PreprocessedImage preprocessed;
                    float[][] embeddings;
                    try {
                        preprocessed = visionModel.preprocessImage(image.encodedBytes());
                        embeddings = visionModel.imageEmbeddings(preprocessed);
                    } catch (java.io.IOException failure) {
                        throw new java.io.UncheckedIOException(failure);
                    }
                    int tokenCount = visionConfig.tokenCount(preprocessed.grid());
                    if (embeddings.length != tokenCount) {
                        throw new IllegalStateException(
                                "vision tower returned "
                                        + embeddings.length
                                        + " rows for grid "
                                        + preprocessed.grid()
                                        + ", which expands to "
                                        + tokenCount);
                    }
                    encodedImages.add(new EncodedImage(preprocessed.grid(), embeddings));
                    parts.add(new ChatFormat.ContentPart.Image(tokenCount));
                }
                default ->
                        throw new IllegalStateException(
                                "a USER message carries no " + piece.getClass().getSimpleName());
            }
        }
        return model.chatFormat().encodeRichMessage(ChatFormat.Role.USER, parts);
    }

    /**
     * Pairs the encoded images, in order, with the image-token runs the format expanded into
     * the finished tokens, and computes their MRoPE positions. A run count or length that
     * disagrees with the grids is an encoder/format defect and fails here, before any token
     * reaches the model.
     */
    private static MultimodalPromptPlan imagePlan(
            List<Integer> tokens, List<EncodedImage> images, FlashNextVisionConfig visionConfig) {
        int imageTokenId = visionConfig.imageTokenId();
        int[] ids = tokens.stream().mapToInt(Integer::intValue).toArray();
        List<MultimodalPromptPlan.ImageSpan> spans = new ArrayList<>();
        int next = 0;
        for (int i = 0; i < ids.length; ) {
            if (ids[i] != imageTokenId) {
                i++;
                continue;
            }
            int start = i;
            while (i < ids.length && ids[i] == imageTokenId) i++;
            if (next >= images.size()) {
                throw new IllegalStateException("more image token runs than encoded images");
            }
            EncodedImage image = images.get(next++);
            int expected = visionConfig.tokenCount(image.grid());
            if (i - start != expected) {
                throw new IllegalStateException(
                        "image token run at " + start + " has length " + (i - start)
                                + " but grid " + image.grid() + " expands to " + expected);
            }
            spans.add(new MultimodalPromptPlan.ImageSpan(start, image.grid(), image.embeddings()));
        }
        if (next != images.size()) {
            throw new IllegalStateException("encoded images without a token run remain");
        }
        return MultimodalPromptPlan.of(ids, spans, imageTokenId, visionConfig.spatialMergeSize());
    }

    /**
     * Encodes the resolved reasoning mode, if one was asked for.
     *
     * <p>After the assistant header, because the control primes the turn the model is about to
     * write. {@link ThinkingMode#DEFAULT} appends nothing, so a family with no reasoning phase and
     * a caller who said nothing are the same case — which is what makes DEFAULT the value that
     * changes nothing.
     *
     * <p>The translation from a mode to tokens is the family's, and it stays here rather than
     * anywhere a caller can see: no thinking token or control string reaches the facade.
     */
    void appendThinkingControl(List<Integer> tokens) {
        if (thinkingMode.isExplicit()) {
            tokens.addAll(
                    model.chatFormat().encodeThinkingControl(thinkingMode == ThinkingMode.ENABLED));
        }
    }

    /**
     * The tool calls in a response, or empty when there are none — including when the response
     * contains tool-shaped text that does not parse.
     *
     * <p>A format that cannot do tool calling returns empty rather than throwing: extraction is
     * asked of every response when tools were supplied, and "this format has no tool calls in it"
     * is a true answer, not an error.
     */
    List<ChatContent.ToolCall> extractToolCalls(String responseText) {
        ChatFormat chatFormat = model.chatFormat();
        if (!chatFormat.supportsToolCalling()) {
            return List.of();
        }
        List<ToolCallExtract> extracted;
        try {
            extracted = chatFormat.extractAllToolCalls(responseText);
        } catch (RuntimeException e) {
            // Malformed tool-shaped output is text, not a failure: the caller gets what the model
            // actually said. Reporting a call that did not parse would have them execute something
            // the model never asked for [A4].
            return List.of();
        }
        List<ChatContent.ToolCall> calls = new ArrayList<>(extracted.size());
        for (ToolCallExtract call : extracted) {
            if (call.name() == null || call.name().isBlank() || call.argumentsJson() == null) {
                continue; // not a valid call; not reported as one
            }
            // The id is optional in the internal extract and required on the facade, because a
            // caller has to match a result back to a call. Formats that do not emit one get a
            // positional identifier rather than a null the caller has to handle.
            String id =
                    call.id()
                            .filter(value -> !value.isBlank())
                            .orElse("call-" + (calls.size() + 1));
            calls.add(new ChatContent.ToolCall(id, call.name(), call.argumentsJson()));
        }
        return List.copyOf(calls);
    }

    /** Whether this model's format can do tool calling at all. */
    boolean supportsTools() {
        return model.chatFormat().supportsToolCalling();
    }

    /** The stop tokens for a request, tool-aware when the request carries tools [§6]. */
    java.util.Set<Integer> stopTokens(boolean withTools) {
        ChatFormat chatFormat = model.chatFormat();
        return withTools ? chatFormat.getToolAwareStopTokens() : chatFormat.getStopTokens();
    }

    private static String textOf(ChatMessage message) {
        return message.content().stream()
                .filter(ChatContent.Text.class::isInstance)
                .map(ChatContent.Text.class::cast)
                .map(ChatContent.Text::text)
                .collect(Collectors.joining());
    }

    /**
     * The tool specifications as the JSON the formats splice in.
     *
     * <p>One JSON object per tool, separated by blank lines — the shape the integrations build and
     * the formats were written against.
     */
    private static String buildToolsJson(List<ToolSpec> tools) {
        return tools.stream()
                .map(ConversationEncoder::toolJson)
                .collect(Collectors.joining("\n\n"));
    }

    /**
     * One tool, as the chat formats expect to read it.
     *
     * <p><b>Compact, and with an empty description omitted</b> — deliberately, because this is the
     * text a model reads and reasons about. The shape and spacing match what the LangChain4j and
     * Quarkus adapters produced before they moved onto this facade (a {@code LinkedHashMap} through
     * their JSON writer), and that is the text the formats' tool prompts were written against.
     * Rendering {@code "description": ""} where there was previously no key at all is exactly the
     * kind of difference a model notices and this project cannot predict.
     */
    private static String toolJson(ToolSpec tool) {
        StringBuilder json =
                new StringBuilder("{\"type\":\"function\",\"function\":{\"name\":\"")
                        .append(escape(tool.name()))
                        .append('"');
        if (!tool.description().isEmpty()) {
            json.append(",\"description\":\"").append(escape(tool.description())).append('"');
        }
        return json.append(",\"parameters\":")
                .append(tool.parametersJsonSchema())
                .append("}}")
                .toString();
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}

package org.beehive.jitllm.server;

import static org.beehive.jitllm.model.loader.ModelLoader.loadModel;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import org.beehive.jitllm.api.ChatContent;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.beehive.jitllm.api.GenerationEvent;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.ReasoningEffort;
import org.beehive.jitllm.api.ThinkingMode;
import org.beehive.jitllm.api.TokenLogprob;
import org.beehive.jitllm.api.ToolSpec;
import org.beehive.jitllm.integration.cli.StartupDiagnostics;
import org.beehive.jitllm.model.Model;

/**
 * OpenAI-compatible HTTP server for jitllm, built on the JDK {@link HttpServer} (no external
 * dependencies). Exposes the loaded model behind the endpoints an OpenAI client already speaks:
 *
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — chat, streaming (SSE) or full JSON.
 *   <li>{@code POST /v1/completions} — text completion (prompt as a single user turn).
 *   <li>{@code GET /v1/models} — the one served model, with the context length it was loaded with,
 *       so a client can size its token budget instead of guessing.
 *   <li>{@code GET /health} — liveness.
 * </ul>
 *
 * <p>Generation is serialized on a single {@link InferenceService} (the GPU is one context); HTTP
 * accept is multi-threaded so clients queue cleanly. Run:
 *
 * <pre>
 *   java. org.beehive.jitllm.server.OpenAIServer --model model.gguf --port 8080 --gpu
 * </pre>
 *
 * <p>{@code --ctx N} sizes the KV cache and therefore the context the server advertises. It
 * defaults to the model's own, which is what {@code ModelOptions} means by 0 and what the loaders
 * already clamp a larger request down to.
 */
public final class OpenAIServer implements AutoCloseable {

    /**
     * Either path, behind one call. The engine-backed service batches several conversations; the
     * original serializes them behind a lock. The HTTP handlers do not need to know which.
     */
    private interface Generator extends AutoCloseable {
        InferenceService.Result generate(
                InferenceService.Request request, java.util.function.Consumer<String> onToken);

        /** As above, with per-token events for streaming logprobs where the path can supply them. */
        default InferenceService.Result generate(
                InferenceService.Request request,
                java.util.function.Consumer<String> onToken,
                java.util.function.Consumer<GenerationEvent> onEvent) {
            return generate(request, onToken);
        }

        default boolean greedyOnly() {
            return false;
        }

        /** Whether this path can return per-token log-probabilities. */
        default boolean supportsLogprobs() {
            return false;
        }

        /** Whether this path honours per-request thinking / reasoning-effort overrides. */
        default boolean supportsReasoningControls() {
            return false;
        }

        @Override
        void close();
    }

    /** Reported when the context length is not known — {@code /v1/models} then omits it. */
    static final int UNKNOWN_CONTEXT_LENGTH = 0;

    private final Generator service;
    private final String servedModel;
    private final boolean gpu;
    private final int contextLength;
    private int port;
    private HttpServer http;
    private java.util.concurrent.ExecutorService httpWorkers;
    private LocalModel ownedModel;

    /** {@code POST /v1/systemone}; null when the serving mode cannot host a decision session. */
    private SystemOneService systemOne;

    private boolean closed;
    private final AtomicLong seq = new AtomicLong();

    public OpenAIServer(InferenceService service, String servedModel, boolean gpu) {
        this(wrap(service), servedModel, gpu, UNKNOWN_CONTEXT_LENGTH);
    }

    /**
     * @param contextLength tokens the model was loaded with, advertised on {@code /v1/models}; 0
     *     when unknown, which omits the field rather than publishing a zero a client would believe
     */
    public OpenAIServer(
            InferenceService service, String servedModel, boolean gpu, int contextLength) {
        this(wrap(service), servedModel, gpu, contextLength);
    }

    /** Engine-backed: concurrent requests share one batch instead of one lock. */
    public OpenAIServer(EngineInferenceService service, String servedModel, boolean gpu) {
        this(wrap(service), servedModel, gpu, UNKNOWN_CONTEXT_LENGTH);
    }

    /** Engine-backed, advertising the context length the batch was sized for. */
    public OpenAIServer(
            EngineInferenceService service, String servedModel, boolean gpu, int contextLength) {
        this(wrap(service), servedModel, gpu, contextLength);
    }

    private static Generator wrap(InferenceService delegate) {
        return new Generator() {
            @Override
            public InferenceService.Result generate(
                    InferenceService.Request request, java.util.function.Consumer<String> onToken) {
                return delegate.generate(request, onToken);
            }

            @Override
            public InferenceService.Result generate(
                    InferenceService.Request request,
                    java.util.function.Consumer<String> onToken,
                    java.util.function.Consumer<GenerationEvent> onEvent) {
                return delegate.generate(request, onToken, onEvent);
            }

            @Override
            public boolean supportsLogprobs() {
                return true;
            }

            @Override
            public boolean supportsReasoningControls() {
                return true;
            }

            @Override
            public void close() {
                delegate.close();
            }
        };
    }

    private static Generator wrap(EngineInferenceService delegate) {
        return new Generator() {
            @Override
            public InferenceService.Result generate(
                    InferenceService.Request request, java.util.function.Consumer<String> onToken) {
                // The engine path never passes through InferenceService, the authoritative
                // counting point, so count its generations here from the Result it returns.
                // The facade adapter above deliberately does not: InferenceService already did.
                InferenceService.Metrics.beginGeneration();
                long startedNanos = System.nanoTime();
                int promptTokens = 0;
                int completionTokens = 0;
                try {
                    InferenceService.Result result = delegate.generate(request, onToken);
                    promptTokens = result.promptTokens();
                    completionTokens = result.completionTokens();
                    return result;
                } finally {
                    InferenceService.Metrics.finishGeneration(
                            System.nanoTime() - startedNanos, promptTokens, completionTokens);
                }
            }

            @Override
            public boolean greedyOnly() {
                return true;
            }

            @Override
            public void close() {
                delegate.close();
            }
        };
    }

    private OpenAIServer(Generator service, String servedModel, boolean gpu, int contextLength) {
        this.service = service;
        this.servedModel = servedModel;
        this.gpu = gpu;
        this.contextLength = contextLength;
    }

    /**
     * The context length to load and advertise, following the rule the model loaders already use: a
     * positive request is honoured but never beyond what the model itself supports, and anything
     * else means the model's own.
     *
     * @param requested the {@code --ctx} value, or 0 when the flag was not given
     * @param modelContextLength the context length the model declares
     */
    /**
     * The loaders' spelling of a requested window: they take {@code -1} for "the model's own", and
     * serve's {@code 0} means the same. Passing 0 through sized a zero-block key/value pool.
     */
    static int loaderContextLength(int requested) {
        return requested > 0 ? requested : -1;
    }

    static int resolveContextLength(int requested, int modelContextLength) {
        return requested > 0 ? Math.min(requested, modelContextLength) : modelContextLength;
    }

    /**
     * Characters per token assumed when checking a prompt against the window.
     *
     * <p>An exact count needs the tokenizer, and this package cannot reach one: the facade path
     * holds a {@link org.beehive.jitllm.api.LocalModel}, whose surface is identity and
     * configuration only. Four is deliberately generous — real text, and code especially, tokenizes
     * to *more* tokens than this predicts — so the estimate errs toward accepting. It therefore
     * catches a prompt that is grossly over the window and lets a marginal one through to the
     * engine, which is the right way round for a guard that must never refuse a request that would
     * have worked.
     */
    static final int ESTIMATED_CHARS_PER_TOKEN = 4;
    /**
     * Generous upper bound on characters per token (measured English averages ~6.5): a prompt
     * rejected by the checks below cannot fit even at this rate, so legitimate near-window
     * prompts are never refused on the heuristic alone; tokenization downstream stays exact.
     */
    static final int MAX_CHARS_PER_TOKEN = 8;

    /**
     * Why this request cannot be served, or {@code null} when it can.
     *
     * <p>Every check here is one that used to be absent, and each absence produced a failure that
     * looked like something else: a model name nobody validated meant a client asking for model A
     * was answered by model B with no indication; a prompt past the window was accepted, ground for
     * hours, and returned a successful response with an empty completion.
     *
     * @param requestedModel the request's {@code model} field, or {@code null} when absent
     * @param servedModel the one model this server loaded
     * @param maxTokens the request's output cap
     * @param promptChars total characters of prompt content
     * @param contextLength the window, or 0 when unknown (every context check is then skipped)
     */
    static String validationError(
            String requestedModel,
            String servedModel,
            int maxTokens,
            int promptChars,
            int contextLength) {
        if (requestedModel != null
                && !requestedModel.isBlank()
                && !requestedModel.equals(servedModel)) {
            return "This server serves '"
                    + servedModel
                    + "', not '"
                    + requestedModel
                    + "'. One model is loaded per process; GET /v1/models reports which.";
        }
        if (contextLength <= 0) {
            return null;
        }
        if (maxTokens >= contextLength) {
            return "max_tokens "
                    + maxTokens
                    + " leaves no room for a prompt in a context of "
                    + contextLength
                    + " tokens.";
        }
        int estimatedPromptTokens = promptChars / ESTIMATED_CHARS_PER_TOKEN;
        int minimumPromptTokens = promptChars / MAX_CHARS_PER_TOKEN;
        if (minimumPromptTokens >= contextLength) {
            return "prompt is about "
                    + estimatedPromptTokens
                    + " tokens ("
                    + promptChars
                    + " characters), which does not fit a context of "
                    + contextLength
                    + " tokens.";
        }
        if (minimumPromptTokens + maxTokens >= contextLength) {
            return "prompt (about "
                    + estimatedPromptTokens
                    + " tokens) plus max_tokens "
                    + maxTokens
                    + " exceeds the context of "
                    + contextLength
                    + " tokens.";
        }
        return null;
    }

    /** A request's parsed {@code logprobs} / {@code top_logprobs} / {@code n}; error when unservable. */
    record LogprobRequest(boolean enabled, int topLogprobs, int n, String error) {

        static LogprobRequest failure(String error) {
            return new LogprobRequest(false, 0, 1, error);
        }
    }

    /** An integral JSON number's value, or {@code null} when it is not one. */
    private static Integer integral(Object value) {
        if (!(value instanceof Number number)) {
            return null;
        }
        double d = number.doubleValue();
        return Double.isFinite(d) && d == Math.rint(d) ? number.intValue() : null;
    }

    /**
     * Parses {@code logprobs}, {@code top_logprobs} and {@code n} into what generation should do.
     *
     * <p>Chat spells {@code logprobs} as a boolean with {@code top_logprobs} (0–20) for the
     * alternatives; completions spell it as the alternatives count itself. Both are served from
     * the facade's per-token log-probabilities on the session path. The continuous-batching
     * engine path exposes none, so there an enabling value is refused rather than dropped — as
     * these fields all were before the facade could supply them. {@code n > 1} is served by
     * generating the choices sequentially on the serialized session (choice <i>i</i> samples with
     * {@code seed + i}); with streaming it stays refused, because the chunk writer serves one
     * choice's token stream at a time.
     */
    static LogprobRequest parseLogprobRequest(
            boolean chat, Object logprobs, Object topLogprobs, Object n,
            boolean stream, boolean supported) {
        int count = 1;
        if (n != null) {
            Integer value = integral(n);
            if (value == null) {
                return LogprobRequest.failure("'n' must be a number.");
            }
            if (value < 1) {
                return LogprobRequest.failure("'n' must be at least 1: " + value);
            }
            count = value;
        }
        if (stream && count > 1) {
            return LogprobRequest.failure(
                    "'n' > 1 is not supported with 'stream': choices are generated sequentially."
                            + " Omit 'stream', or set 'n' to 1 and issue one request per sample.");
        }
        Integer top = null;
        if (topLogprobs != null) {
            top = integral(topLogprobs);
            if (top == null) {
                return LogprobRequest.failure("'top_logprobs' must be a number.");
            }
            if (top < 0 || top > 20) {
                return LogprobRequest.failure("'top_logprobs' must be in [0, 20]: " + top);
            }
        }
        boolean enabled = false;
        int alternatives = 0;
        if (chat) {
            if (logprobs != null && !(logprobs instanceof Boolean)) {
                return LogprobRequest.failure("'logprobs' must be a boolean for chat completions.");
            }
            enabled = Boolean.TRUE.equals(logprobs);
            if (top != null) {
                alternatives = top;
            }
        } else if (logprobs instanceof Boolean flag) {
            enabled = flag;
            if (top != null) {
                alternatives = top;
            }
        } else if (logprobs instanceof Number) {
            Integer value = integral(logprobs);
            if (value == null || value < 0 || value > 20) {
                return LogprobRequest.failure(
                        "'logprobs' must be a count in [0, 20] for completions: " + logprobs);
            }
            enabled = true;
            alternatives = top == null ? value : top;
        } else if (logprobs != null) {
            return LogprobRequest.failure(
                    "'logprobs' must be a boolean or a count for completions.");
        } else if (top != null) {
            alternatives = top;
        }
        if (alternatives > 0 && !enabled) {
            return LogprobRequest.failure(
                    "'top_logprobs' requires 'logprobs' to be enabled (true for chat, a count"
                            + " for completions).");
        }
        if (enabled && !supported) {
            return LogprobRequest.failure(
                    "'logprobs' is not supported on the continuous-batching serving path: it"
                            + " does not expose per-token probabilities. Restart without"
                            + " --continuous-batching, or omit 'logprobs'.");
        }
        return new LogprobRequest(enabled, alternatives, count, null);
    }

    /** A request's parsed per-request reasoning controls; either may be null (inherit). */
    record ReasoningControls(
            ThinkingMode thinkingMode, ReasoningEffort reasoningEffort, String error) {

        static ReasoningControls failure(String error) {
            return new ReasoningControls(null, null, error);
        }

        boolean any() {
            return thinkingMode != null || reasoningEffort != null;
        }
    }

    /**
     * Parses {@code chat_template_kwargs.enable_thinking} and {@code reasoning_effort} into the
     * facade's per-request overrides.
     *
     * <p>These take effect per request on the session path: the request's mode and effort
     * override the session's resolved values for that request's encoding only. A value the
     * loaded model's format cannot express (an explicit mode on a model with no reasoning phase,
     * an effort on a template without the control) is rejected by the facade at generation and
     * reported as a 400 by the non-streaming handler. {@code reasoning_effort: "high"} maps to
     * {@code XHIGH}, as the Qwen template treats the two alike; {@code "none"} disables thinking
     * and {@code "minimal"} is {@code LOW}. The engine path primes its own prompt and has no
     * such controls, so there any explicit value is refused.
     */
    static ReasoningControls parseReasoningControls(
            Object chatTemplateKwargs, Object reasoningEffort, boolean supported) {
        ThinkingMode thinking = null;
        if (chatTemplateKwargs != null) {
            if (!(chatTemplateKwargs instanceof Map<?, ?> kwargs)) {
                return ReasoningControls.failure("'chat_template_kwargs' must be an object.");
            }
            Object value = kwargs.get("enable_thinking");
            if (value != null) {
                if (!(value instanceof Boolean flag)) {
                    return ReasoningControls.failure(
                            "'chat_template_kwargs.enable_thinking' must be a boolean.");
                }
                thinking = flag ? ThinkingMode.ENABLED : ThinkingMode.DISABLED;
            }
        }
        ReasoningEffort effort = null;
        if (reasoningEffort != null) {
            if (!(reasoningEffort instanceof String text) || text.isBlank()) {
                return ReasoningControls.failure("'reasoning_effort' must be a non-empty string.");
            }
            switch (text.strip().toLowerCase(java.util.Locale.ROOT)) {
                case "low", "minimal" -> effort = ReasoningEffort.LOW;
                case "medium" -> effort = ReasoningEffort.MEDIUM;
                case "high", "xhigh" -> effort = ReasoningEffort.XHIGH;
                case "none" -> {
                    if (thinking == ThinkingMode.ENABLED) {
                        return ReasoningControls.failure(
                                "'reasoning_effort': \"none\" contradicts"
                                        + " 'chat_template_kwargs.enable_thinking': true.");
                    }
                    thinking = ThinkingMode.DISABLED;
                }
                default -> {
                    return ReasoningControls.failure(
                            "'reasoning_effort' value '"
                                    + text
                                    + "' is not recognised; supported values are none, low,"
                                    + " medium, high and xhigh.");
                }
            }
        }
        if (thinking == ThinkingMode.DISABLED && effort != null) {
            return ReasoningControls.failure(
                    "'reasoning_effort' cannot be combined with thinking disabled"
                            + " ('chat_template_kwargs.enable_thinking': false): there is no"
                            + " reasoning phase for it to apply to.");
        }
        ReasoningControls controls = new ReasoningControls(thinking, effort, null);
        if (controls.any() && !supported) {
            return ReasoningControls.failure(
                    "Per-request reasoning controls ('enable_thinking' / 'reasoning_effort') are"
                            + " not supported on the continuous-batching serving path. Restart"
                            + " without --continuous-batching, or omit them.");
        }
        return controls;
    }

    /**
     * One choice's {@code logprobs} object in OpenAI shape: a {@code content} entry per
     * generated token with its text, log-probability, UTF-8 bytes and top alternatives.
     *
     * <p>A log-probability whose probability underflowed to zero arrives as {@code -Infinity},
     * which JSON cannot carry; it is written as {@code -9999}, the value OpenAI itself uses for
     * vanishingly unlikely tokens. {@code bytes} is the token text's UTF-8, or {@code null} when
     * the text holds a replacement character (an alternative token carrying part of a multi-byte
     * character on its own): its raw bytes are not recoverable from the text, and {@code null}
     * is the shape OpenAI gives that case.
     */
    static Map<String, Object> logprobsPayload(List<TokenLogprob> entries) {
        List<Object> content = new ArrayList<>(entries.size());
        for (TokenLogprob entry : entries) {
            List<Object> tops = new ArrayList<>(entry.topLogprobs().size());
            for (TokenLogprob.Alternative alternative : entry.topLogprobs()) {
                Map<String, Object> top = new LinkedHashMap<>();
                top.put("token", alternative.text());
                top.put("logprob", wireLogprob(alternative.logprob()));
                top.put("bytes", wireBytes(alternative.text()));
                tops.add(top);
            }
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("token", entry.text());
            item.put("logprob", wireLogprob(entry.logprob()));
            item.put("bytes", wireBytes(entry.text()));
            item.put("top_logprobs", tops);
            content.add(item);
        }
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("content", content);
        return payload;
    }

    private static double wireLogprob(double logprob) {
        return Double.isFinite(logprob) ? logprob : -9999.0;
    }

    private static List<Integer> wireBytes(String text) {
        if (text.indexOf('�') >= 0) {
            return null;
        }
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> bytes = new ArrayList<>(utf8.length);
        for (byte b : utf8) {
            bytes.add(b & 0xFF);
        }
        return bytes;
    }

    /** A generation split into its reasoning phase and its answer; reasoning null when absent. */
    record ReasoningSplit(String reasoningContent, String content) {}

    private static final String THINK_OPEN = "<think>";
    private static final String THINK_CLOSE = "</think>";

    /**
     * Splits a leading {@code <think>...</think>} block out of generated text, the HTTP-layer
     * equivalent of staged SGLang's {@code --reasoning-parser qwen3}.
     *
     * <p>The markup is really there to split: on the facade path nothing strips it. {@code
     * DelegatingSession} returns {@code events.finish()} verbatim as {@code GenerationResult}
     * text, and with thinking enabled the Qwen formats encode no control (the generation prompt
     * does not open the block either), so the model writes {@code <think>} itself as its first
     * output. With thinking disabled the pre-closed block is encoded in the <em>prompt</em>, so
     * the output is a plain answer with no tags at all.
     *
     * <p>Only a block that opens the text is split; anything else — no tags, or tags quoted
     * mid-answer — is returned byte-for-byte with {@code reasoningContent} null, so a
     * non-reasoning response is unchanged. An unclosed block (generation stopped at the token
     * cap mid-reasoning) makes everything after the tag reasoning and the content empty, as the
     * qwen3 parsers report it. The newlines the template puts around the block are separators,
     * not content: leading/trailing {@code \r\n} around the reasoning and leading {@code \r\n}
     * before the answer are stripped, and nothing else is touched.
     */
    static ReasoningSplit splitReasoning(String text) {
        if (text == null) {
            return new ReasoningSplit(null, "");
        }
        if (!text.startsWith(THINK_OPEN)) {
            return new ReasoningSplit(null, text);
        }
        int close = text.indexOf(THINK_CLOSE, THINK_OPEN.length());
        if (close < 0) {
            return new ReasoningSplit(stripLeadingNewlines(text.substring(THINK_OPEN.length())), "");
        }
        String reasoning =
                stripTrailingNewlines(
                        stripLeadingNewlines(text.substring(THINK_OPEN.length(), close)));
        String content = stripLeadingNewlines(text.substring(close + THINK_CLOSE.length()));
        return new ReasoningSplit(reasoning, content);
    }

    private static String stripLeadingNewlines(String text) {
        int start = 0;
        while (start < text.length() && (text.charAt(start) == '\n' || text.charAt(start) == '\r')) {
            start++;
        }
        return text.substring(start);
    }

    private static String stripTrailingNewlines(String text) {
        int end = text.length();
        while (end > 0 && (text.charAt(end - 1) == '\n' || text.charAt(end - 1) == '\r')) {
            end--;
        }
        return text.substring(0, end);
    }

    /**
     * The streaming form of {@link #splitReasoning}: token pieces in, classified deltas out.
     *
     * <p>Tags may be split across pieces, so a piece is only emitted once no suffix of it could
     * still grow into the tag that would reclassify it: at the start, anything that is still a
     * prefix of {@code <think>} is held; inside the block, a suffix that is a prefix of {@code
     * </think>} is held. Text that does not open with {@code <think>} is content from the first
     * decidable piece and is never reclassified, matching the non-streaming split: concatenating
     * the reasoning deltas and the content deltas separately reproduces {@link #splitReasoning}
     * of the whole text (trailing separator newlines are held with the tag prefix, above).
     */
    static final class ReasoningStreamParser {

        /** One classified piece of streamed output. */
        record Delta(boolean reasoning, String text) {}

        private enum Mode {
            START,
            REASONING,
            CONTENT
        }

        private Mode mode = Mode.START;
        private final StringBuilder pending = new StringBuilder();
        private boolean stripLeadingNewlines;

        List<Delta> feed(String piece) {
            List<Delta> out = new ArrayList<>();
            if (piece != null && !piece.isEmpty()) {
                pending.append(piece);
                drain(out, false);
            }
            return out;
        }

        /** Everything still held back, classified by where the stream ended. */
        List<Delta> finish() {
            List<Delta> out = new ArrayList<>();
            drain(out, true);
            return out;
        }

        private void drain(List<Delta> out, boolean endOfStream) {
            while (true) {
                switch (mode) {
                    case START -> {
                        String held = pending.toString();
                        if (!endOfStream
                                && held.length() < THINK_OPEN.length()
                                && THINK_OPEN.startsWith(held)) {
                            return; // could still become <think>; wait for more
                        }
                        if (held.startsWith(THINK_OPEN)) {
                            pending.delete(0, THINK_OPEN.length());
                            mode = Mode.REASONING;
                            stripLeadingNewlines = true;
                        } else {
                            mode = Mode.CONTENT;
                        }
                    }
                    case REASONING -> {
                        if (!dropLeadingNewlines() && !endOfStream) {
                            return;
                        }
                        int close = pending.indexOf(THINK_CLOSE);
                        if (close >= 0) {
                            emit(out, true, stripTrailingNewlines(pending.substring(0, close)));
                            pending.delete(0, close + THINK_CLOSE.length());
                            mode = Mode.CONTENT;
                            stripLeadingNewlines = true;
                        } else if (endOfStream) {
                            emit(out, true, pending.toString());
                            pending.setLength(0);
                            return;
                        } else {
                            // Hold a trailing tag prefix, and any newlines right before it:
                            // until the next piece decides, they may be the separator in
                            // front of the closing tag rather than reasoning text.
                            int held = heldTagSuffix(pending, THINK_CLOSE);
                            while (held < pending.length()
                                    && (pending.charAt(pending.length() - held - 1) == '\n'
                                            || pending.charAt(pending.length() - held - 1) == '\r')) {
                                held++;
                            }
                            int emitUpTo = pending.length() - held;
                            emit(out, true, pending.substring(0, emitUpTo));
                            pending.delete(0, emitUpTo);
                            return;
                        }
                    }
                    case CONTENT -> {
                        if (!dropLeadingNewlines() && !endOfStream) {
                            return;
                        }
                        emit(out, false, pending.toString());
                        pending.setLength(0);
                        return;
                    }
                }
            }
        }

        /**
         * Drops separator newlines at the head of a phase; false when nothing but newlines has
         * arrived yet, so the next piece is still subject to the drop.
         */
        private boolean dropLeadingNewlines() {
            if (!stripLeadingNewlines) {
                return true;
            }
            int start = 0;
            while (start < pending.length()
                    && (pending.charAt(start) == '\n' || pending.charAt(start) == '\r')) {
                start++;
            }
            pending.delete(0, start);
            if (pending.isEmpty()) {
                return false;
            }
            stripLeadingNewlines = false;
            return true;
        }

        /** How much of {@code text}'s suffix could still grow into {@code tag}. */
        private static int heldTagSuffix(StringBuilder text, String tag) {
            int max = Math.min(text.length(), tag.length() - 1);
            for (int length = max; length > 0; length--) {
                boolean matches = true;
                for (int i = 0; i < length && matches; i++) {
                    matches = text.charAt(text.length() - length + i) == tag.charAt(i);
                }
                if (matches) {
                    return length;
                }
            }
            return 0;
        }

        private static void emit(List<Delta> out, boolean reasoning, String text) {
            if (!text.isEmpty()) {
                out.add(new Delta(reasoning, text));
            }
        }
    }

    /**
     * One line describing an inbound completion request.
     *
     * <p>The server used to log nothing per request, which made a client's behaviour invisible:
     * whether a prompt ever arrived, which model name it asked for, how large it was, and whether a
     * reply was a rejection or a real generation all had to be inferred from the outside.
     *
     * @param id the response id this request will carry, so a log line and a reply can be paired
     */
    static String requestSummary(
            String id, String path, String model, int promptChars, int maxTokens, boolean stream) {
        return "[req "
                + id
                + "] "
                + path
                + " model="
                + (model == null || model.isBlank() ? "(unset)" : model)
                + " promptChars="
                + promptChars
                + " maxTokens="
                + maxTokens
                + (stream ? " stream" : "");
    }

    /**
     * Total characters of message text, the input to the prompt-size estimate.
     *
     * <p>Only {@link ChatContent.Text} carries characters a tokenizer would see here; a tool call
     * or result contributes its JSON through a different path and is not counted.
     */
    static int promptCharacters(List<ChatMessage> messages) {
        int total = 0;
        for (ChatMessage m : messages) {
            if (m == null) {
                continue;
            }
            for (ChatContent piece : m.content()) {
                if (piece instanceof ChatContent.Text t && t.text() != null) {
                    total += t.text().length();
                }
            }
        }
        return total;
    }

    /**
     * The {@code /v1/models} body: the OpenAI model shape plus {@code context_length}.
     *
     * <p>OpenAI does not specify a context field, so clients read one of a handful of de-facto
     * spellings; {@code context_length} is the one OpenRouter established and the widest set of
     * OpenAI-compatible clients already look for.
     */
    static Map<String, Object> modelsPayload(String servedModel, int contextLength) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", servedModel);
        entry.put("object", "model");
        entry.put("created", 0);
        entry.put("owned_by", "jitllm");
        if (contextLength > 0) {
            entry.put("context_length", contextLength);
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("object", "list");
        body.put("data", List.of(entry));
        return body;
    }

    public static void main(String[] args) throws IOException {
        org.beehive.jitllm.integration.cli.CliErrors.reportDiagnostics(() -> run(args));
    }

    private static void run(String[] args) throws IOException {
        if (java.util.Arrays.asList(args).contains("--help")
                || java.util.Arrays.asList(args).contains("-h")) {
            System.out.println(
                    "Usage: OpenAIServer [serve] --model FILE [--ctx-size|--ctx N (default: model's own)] [--host 127.0.0.1] [--port 8080] [--gpu] [--continuous-batching SLOTS (experimental)] [-v]");
            return;
        }
        ServerOptions options = ServerOptions.parse(args);
        var config = options.model();
        // HTTP requests choose sampling independently; a greedy-only device sampler cannot be
        // pinned globally.
        System.clearProperty("jitllm.deviceSample");
        long startedNs = System.nanoTime();
        Path path = config.model();
        String served = path.getFileName().toString().replaceAll("\\.gguf$", "");
        ModelOptions modelOptions = config.modelOptions();
        OpenAIServer server;
        if (options.continuousBatching()) {
            System.err.println(ServerOptions.CONTINUOUS_BATCHING_WARNING);
            Model model =
                    loadModel(
                            path, loaderContextLength(config.contextLength()), true, config.gpu());
            long loadNs = System.nanoTime() - startedNs;
            // The engine needs a concrete window; the facade path reads its own back after load.
            int contextLength =
                    resolveContextLength(
                            config.contextLength(), model.configuration().contextLength());
            if (model.weights().dataType() != org.beehive.jitllm.runtime.tensor.DataType.F16) {
                throw new IllegalArgumentException(
                        "Continuous batching requires FP16 Llama/Qwen3 weights");
            }
            EngineInferenceService service =
                    new EngineInferenceService(
                            model,
                            options.batchSlots(),
                            options.maxQueuedRequests(),
                            contextLength,
                            options.prefixCacheEntries());
            server = new OpenAIServer(service, served, config.gpu(), contextLength);
            try {
                if (StartupDiagnostics.verbose()) {
                    System.err.print(
                            StartupDiagnostics.render(
                                    model,
                                    path,
                                    service.executionInfo(),
                                    "greedy (continuous batching)",
                                    modelOptions,
                                    loadNs,
                                    startedNs));
                }
                server.start(options.host(), options.port());
            } catch (IOException | RuntimeException | Error failure) {
                server.close();
                throw failure;
            }
        } else {
            // One more session than the chat path uses: /v1/systemone scores on its own session, so
            // a decision request never rewinds a conversation the chat endpoint is in the middle
            // of.
            LocalModel model = LocalModels.load(path, withExtraSession(modelOptions));
            long loadNs = System.nanoTime() - startedNs;
            InferenceService service;
            try {
                service = new InferenceService(model);
            } catch (RuntimeException | Error failure) {
                model.close();
                throw failure;
            }
            server = new OpenAIServer(service, served, config.gpu(), model.info().contextLength());
            server.ownedModel = model;
            if (model instanceof org.beehive.jitllm.api.TextGenerationModel generator) {
                try {
                    server.systemOne = new SystemOneService(generator, served);
                } catch (RuntimeException unavailable) {
                    System.err.println(
                            "jitllm serve: /v1/systemone disabled: " + unavailable.getMessage());
                }
            }
            try {
                if (StartupDiagnostics.verbose()) {
                    System.err.print(
                            StartupDiagnostics.render(
                                    model,
                                    service.prepare(),
                                    "per HTTP request",
                                    modelOptions,
                                    loadNs,
                                    startedNs));
                }
                server.start(options.host(), options.port());
            } catch (IOException | RuntimeException | Error failure) {
                server.close();
                throw failure;
            }
        }
        if (StartupDiagnostics.verbose()) {
            System.err.println(
                    "[server] context per request="
                            + server.contextLength
                            + " continuousBatchingSlots="
                            + options.batchSlots()
                            + " maxQueuedRequests="
                            + options.maxQueuedRequests());
        }
        Runtime.getRuntime().addShutdownHook(new Thread(server::close, "server-shutdown"));
    }

    public void start(int port) throws IOException {
        start("127.0.0.1", port);
    }

    public synchronized void start(String host, int port) throws IOException {
        if (http != null || closed)
            throw new IllegalStateException("Server already started or closed");
        http = HttpServer.create(new InetSocketAddress(host, port), 0);
        this.port = http.getAddress().getPort();
        http.createContext("/", this::handleIndex);
        http.createContext("/health", this::handleHealth);
        http.createContext("/metrics", this::handleMetrics);
        http.createContext("/v1/models", this::handleModels);
        http.createContext("/v1/chat/completions", ex -> handleCompletion(ex, true));
        http.createContext("/v1/completions", ex -> handleCompletion(ex, false));
        http.createContext("/v1/systemone", this::handleSystemOne);
        httpWorkers = Executors.newFixedThreadPool(8);
        http.setExecutor(httpWorkers);
        http.start();
        String bound = http.getAddress().getAddress().getHostAddress();
        if (bound.contains(":")) bound = "[" + bound + "]";
        System.err.println(
                "[server] listening on http://"
                        + bound
                        + ":"
                        + this.port
                        + "  model="
                        + servedModel
                        + (contextLength > 0 ? "  ctx=" + contextLength : ""));
    }

    public synchronized int port() {
        return port;
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        if (http != null) http.stop(1);
        if (httpWorkers != null) httpWorkers.shutdownNow();
        try {
            if (systemOne != null) systemOne.close();
            service.close();
        } finally {
            if (ownedModel != null) ownedModel.close();
        }
    }

    // ── Endpoints ─────────────────────────────────────────────────────────────

    private void handleIndex(HttpExchange ex) throws IOException {
        if (!"/".equals(ex.getRequestURI().getPath())) {
            sendError(ex, 404, "Not found");
            return;
        }
        byte[] bytes =
                INDEX_HTML
                        .replace("{{model}}", servedModel)
                        .replace("{{backend}}", gpu ? "GPU (TornadoVM)" : "CPU")
                        .replace("{{port}}", String.valueOf(port))
                        .getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static final String INDEX_HTML =
            """
            <!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <title>jitllm server</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                     max-width: 780px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #1a1a1a; }
              h1 { font-size: 1.5rem; margin-bottom: 0; }
              .sub { color: #666; margin-top: 4px; }
              .badges span { display: inline-block; background: #eee; border-radius: 4px; padding: 2px 8px;
                             margin: 4px 6px 0 0; font-size: 0.85rem; }
              table { border-collapse: collapse; margin: 12px 0; }
              td { padding: 3px 10px 3px 0; vertical-align: top; }
              td.k { color: #666; }
              code, pre { background: #f4f4f4; border-radius: 4px; }
              code { padding: 1px 5px; }
              pre { padding: 10px; overflow-x: auto; }
              a { color: #0969da; }
              hr { border: none; border-top: 1px solid #eee; margin: 24px 0; }
              ul { padding-left: 20px; }
            </style>
            </head>
            <body>
              <h1>jitllm</h1>
              <div class="sub">Local OpenAI-compatible inference server</div>

              <table>
                <tr><td class="k">Model</td><td><code>{{model}}</code></td></tr>
                <tr><td class="k">Backend</td><td>{{backend}}</td></tr>
                <tr><td class="k">Port</td><td>{{port}}</td></tr>
              </table>

              <p>
                This is a local instance of
                <a href="https://github.com/beehive-lab/jitllm" target="_blank">jitllm</a>,
                a Llama3-family inference engine written in native Java and automatically accelerated on
                GPUs with <a href="https://github.com/beehive-lab/TornadoVM" target="_blank">TornadoVM</a>.
                It supports Llama3, Mistral, Devstral 2, Qwen2.5, Qwen3, Phi-3, IBM Granite 3.2+, and
                IBM Granite 4.0 models in GGUF format, and is also used as the GPU inference engine behind
                the <a href="https://docs.quarkiverse.io/quarkus-langchain4j/dev/gpullama3-chat-model.html" target="_blank">Quarkus</a>
                and <a href="https://docs.langchain4j.dev/integrations/language-models/gpullama3-java" target="_blank">LangChain4j</a>
                integrations.
              </p>

              <hr>

              <h3>API endpoints</h3>
              <ul>
                <li><code>GET  /health</code> — liveness check</li>
                <li><code>GET  /v1/models</code> — list the served model</li>
                <li><code>POST /v1/chat/completions</code> — chat completions (supports <code>"stream": true</code> SSE)</li>
                <li><code>POST /v1/completions</code> — text completions (supports <code>"stream": true</code> SSE)</li>
              </ul>

              <h3>Quick test</h3>
              <pre><code>curl http://localhost:{{port}}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "{{model}}",
            "messages": [{"role": "user", "content": "Hello!"}]
          }'</code></pre>

              <p class="sub">
                Any OpenAI-compatible client (e.g. the <code>openai</code> Python/Node SDK) can point its
                base URL at this server. Generation runs on a single serialized GPU/CPU context, so
                concurrent requests are queued and processed one at a time.
              </p>
            </body>
            </html>
            """;

    private void handleHealth(HttpExchange ex) throws IOException {
        InferenceService.Metrics.recordRequest("/health");
        sendJson(ex, 200, Map.of("status", "ok"));
    }

    /** Prometheus text exposition; the counters live in {@link InferenceService.Metrics}. */
    private void handleMetrics(HttpExchange ex) throws IOException {
        if (!"GET".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed — use GET");
            return;
        }
        byte[] bytes = InferenceService.Metrics.render().getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void handleModels(HttpExchange ex) throws IOException {
        InferenceService.Metrics.recordRequest("/v1/models");
        System.err.println(
                "[req] GET /v1/models -> "
                        + servedModel
                        + (contextLength > 0 ? " ctx=" + contextLength : " ctx=unknown"));
        sendJson(ex, 200, modelsPayload(servedModel, contextLength));
    }

    @SuppressWarnings("unchecked")
    private void handleCompletion(HttpExchange ex, boolean chat) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed — use POST");
            return;
        }
        Map<String, Object> body;
        try {
            String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            body = Json.parseObject(raw);
        } catch (Exception e) {
            sendError(ex, 400, "Invalid JSON body: " + e.getMessage());
            return;
        }

        List<ChatMessage> messages = new ArrayList<>();
        try {
            if (chat) {
                Object msgs = body.get("messages");
                if (!(msgs instanceof List) || ((List<?>) msgs).isEmpty()) {
                    sendError(ex, 400, "'messages' must be a non-empty array");
                    return;
                }
                var toolNames = new HashMap<String, String>();
                for (Object o : (List<Object>) msgs) {
                    Map<String, Object> m = (Map<String, Object>) o;
                    String role = Json.str(m, "role", "user");
                    var chatRole = chatRole(role);
                    List<ChatContent> richPieces = null;
                    String content;
                    if (chatRole == ChatRole.USER && m.get("content") instanceof List) {
                        richPieces = contentPieces(m.get("content"));
                        content = textOf(richPieces);
                    } else {
                        content = messageText(m.get("content"));
                    }
                    if (chatRole == ChatRole.TOOL) {
                        String callId = Json.str(m, "tool_call_id", null);
                        if (callId == null || callId.isBlank())
                            throw new IllegalArgumentException("a tool message needs tool_call_id");
                        String name = Json.str(m, "name", toolNames.get(callId));
                        if (name == null || name.isBlank())
                            throw new IllegalArgumentException(
                                    "a tool message needs a name or an earlier assistant tool_call with its id");
                        messages.add(new ChatMessage(chatRole,
                                List.of(new ChatContent.ToolResult(callId, name, content))));
                    } else if (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty()) {
                        var pieces = new ArrayList<ChatContent>();
                        if (!content.isEmpty()) pieces.add(new ChatContent.Text(content));
                        for (Object c : calls) {
                            Map<String, Object> call = (Map<String, Object>) c;
                            Map<String, Object> fn = (Map<String, Object>) call.get("function");
                            if (fn == null) throw new IllegalArgumentException("a tool_call needs a function object");
                            String id = Json.str(call, "id", "call_" + pieces.size());
                            String name = Json.str(fn, "name", null);
                            if (name == null || name.isBlank())
                                throw new IllegalArgumentException("a tool_call needs a function name");
                            Object arguments = fn.get("arguments");
                            pieces.add(new ChatContent.ToolCall(id, name,
                                    arguments instanceof String s ? s : Json.write(arguments == null ? Map.of() : arguments)));
                            toolNames.put(id, name);
                        }
                        messages.add(new ChatMessage(chatRole, pieces));
                    } else if (richPieces != null
                            && richPieces.stream().anyMatch(p -> p instanceof ChatContent.Image)) {
                        messages.add(new ChatMessage(chatRole, richPieces));
                    } else {
                        messages.add(ChatMessage.of(chatRole, content));
                    }
                }
            } else {
                Object prompt = body.get("prompt");
                String text =
                        prompt instanceof String s ? s : prompt == null ? "" : prompt.toString();
                if (text.isEmpty()) {
                    sendError(ex, 400, "'prompt' must be a non-empty string");
                    return;
                }
                messages.add(ChatMessage.of(ChatRole.USER, text));
            }
        } catch (Exception e) {
            sendError(ex, 400, "Malformed request: " + e.getMessage());
            return;
        }

        int maxTokens =
                Json.intVal(
                        body,
                        "max_tokens",
                        chat ? Json.intVal(body, "max_completion_tokens", 256) : 256);
        float temperature = (float) Json.num(body, "temperature", 0.0);
        float topP = (float) Json.num(body, "top_p", 0.95);
        long seed = (long) Json.num(body, "seed", 1234);
        int topK = Json.intVal(body, "top_k", 0);
        float minP = (float) Json.num(body, "min_p", 0.0);
        float frequencyPenalty = (float) Json.num(body, "frequency_penalty", 0.0);
        float presencePenalty = (float) Json.num(body, "presence_penalty", 0.0);
        float repetitionPenalty = (float) Json.num(body, "repetition_penalty", 1.0);
        List<ToolSpec> tools;
        try {
            tools = parseTools(body);
        } catch (Exception e) {
            sendError(ex, 400, "Malformed request: " + e.getMessage());
            return;
        }
        boolean stream = Json.bool(body, "stream", false);
        LogprobRequest logprobs =
                parseLogprobRequest(
                        chat,
                        body.get("logprobs"),
                        body.get("top_logprobs"),
                        body.get("n"),
                        stream,
                        service.supportsLogprobs());
        ReasoningControls reasoning =
                parseReasoningControls(
                        body.get("chat_template_kwargs"),
                        body.get("reasoning_effort"),
                        service.supportsReasoningControls());

        String requestedModel = Json.str(body, "model", null);
        int promptChars = promptCharacters(messages);
        String path = chat ? "POST /v1/chat/completions" : "POST /v1/completions";

        var req = new InferenceService.Request(messages, maxTokens, temperature, topP, seed,
                topK, minP, frequencyPenalty, presencePenalty, repetitionPenalty, tools,
                reasoning.thinkingMode(), reasoning.reasoningEffort(),
                logprobs.enabled(), logprobs.topLogprobs());
        String id = (chat ? "chatcmpl-" : "cmpl-") + seq.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        String summary = requestSummary(id, path, requestedModel, promptChars, maxTokens, stream);

        String rejection =
                validationError(requestedModel, servedModel, maxTokens, promptChars, contextLength);
        if (rejection == null && service.greedyOnly() && temperature != 0.0f) {
            rejection = "Continuous batching currently requires temperature=0 (greedy sampling)";
        }
        if (rejection == null) {
            rejection = logprobs.error();
        }
        if (rejection == null) {
            rejection = reasoning.error();
        }
        if (rejection != null) {
            System.err.println(summary + " -> 400 " + rejection);
            sendError(ex, 400, rejection);
            return;
        }
        InferenceService.Metrics.recordRequest(chat ? "/v1/chat/completions" : "/v1/completions");

        System.err.println(summary);
        long startNanos = System.nanoTime();
        try {
            if (stream) {
                streamResponse(ex, req, id, created, chat, logprobs);
            } else {
                fullResponse(ex, req, id, created, chat, logprobs);
            }
        } finally {
            // Generation is serialized, so this covers queue wait as well as the work itself —
            // which is the number that explains a slow reply in a client.
            System.err.printf(
                    "[req %s] done in %.1fs%n", id, (System.nanoTime() - startNanos) / 1e9);
        }
    }

    /**
     * The OpenAI role names, mapped onto the facade's roles. An unknown role is a client error, not
     * a silent downgrade to "user": a request whose role the server does not understand would be
     * rendered into the wrong place in the chat template.
     */
    /** User content parts as facade pieces: {@code text} parts and {@code image_url} images. */
    static List<ChatContent> contentPieces(Object content) {
        if (content == null) return List.of(new ChatContent.Text(""));
        if (content instanceof String s) return List.of(new ChatContent.Text(s));
        if (content instanceof List<?> parts) {
            var pieces = new ArrayList<ChatContent>();
            for (Object part : parts) {
                Map<String, Object> piece = (Map<String, Object>) part;
                String type = Json.str(piece, "type", "text");
                switch (type) {
                    case "text" -> pieces.add(new ChatContent.Text(Json.str(piece, "text", "")));
                    case "image_url" -> {
                        Object image = piece.get("image_url");
                        Object url = image instanceof Map<?, ?> map ? map.get("url") : image;
                        if (!(url instanceof String s) || s.isBlank())
                            throw new IllegalArgumentException("an image_url part needs a url");
                        pieces.add(new ChatContent.Image(imageBytes(s)));
                    }
                    default -> throw new IllegalArgumentException(
                            "content part type '" + type + "' is not supported yet");
                }
            }
            return pieces;
        }
        throw new IllegalArgumentException("message content must be a string or an array of parts");
    }

    /** The joined text of already-parsed content pieces (images contribute nothing). */
    static String textOf(List<ChatContent> pieces) {
        var text = new StringBuilder();
        for (ChatContent piece : pieces)
            if (piece instanceof ChatContent.Text t) text.append(t.text());
        return text.toString();
    }

    /** Encoded image bytes from a {@code data:} base64 URL or an http(s) URL, size-capped. */
    static byte[] imageBytes(String url) {
        try {
            if (url.startsWith("data:")) {
                int comma = url.indexOf(',');
                if (comma < 0 || !url.substring(0, comma).contains("base64"))
                    throw new IllegalArgumentException("only base64 data: image URLs are supported");
                byte[] bytes = java.util.Base64.getMimeDecoder().decode(url.substring(comma + 1));
                if (bytes.length > ChatContent.Image.MAX_ENCODED_BYTES)
                    throw new IllegalArgumentException("image exceeds the 32 MiB limit");
                return bytes;
            }
            var uri = java.net.URI.create(url);
            if (!"http".equals(uri.getScheme()) && !"https".equals(uri.getScheme()))
                throw new IllegalArgumentException("image_url must be a data: or http(s) URL");
            var request = java.net.http.HttpRequest.newBuilder(uri)
                    .timeout(java.time.Duration.ofSeconds(15)).GET().build();
            var response = java.net.http.HttpClient.newBuilder()
                    .connectTimeout(java.time.Duration.ofSeconds(5))
                    .followRedirects(java.net.http.HttpClient.Redirect.NORMAL).build()
                    .send(request, java.net.http.HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() / 100 != 2)
                throw new IllegalArgumentException("image fetch failed with HTTP " + response.statusCode());
            try (var in = response.body()) {
                byte[] bytes = in.readNBytes(ChatContent.Image.MAX_ENCODED_BYTES + 1);
                if (bytes.length > ChatContent.Image.MAX_ENCODED_BYTES)
                    throw new IllegalArgumentException("image exceeds the 32 MiB limit");
                return bytes;
            }
        } catch (IllegalArgumentException bad) {
            throw bad;
        } catch (Exception failure) {
            throw new IllegalArgumentException("could not load image: " + failure.getMessage());
        }
    }

    /** Message content as plain text: a string, or the joined {@code text} content parts. */
    static String messageText(Object content) {
        if (content == null) return "";
        if (content instanceof String s) return s;
        if (content instanceof List<?> parts) {
            var text = new StringBuilder();
            for (Object part : parts) {
                Map<String, Object> piece = (Map<String, Object>) part;
                String type = Json.str(piece, "type", "text");
                if (!type.equals("text"))
                    throw new IllegalArgumentException("content part type '" + type + "' is not supported yet");
                text.append(Json.str(piece, "text", ""));
            }
            return text.toString();
        }
        throw new IllegalArgumentException("message content must be a string or an array of parts");
    }

    /** The request's OpenAI {@code tools}/{@code tool_choice} as facade tool specs. */
    static List<ToolSpec> parseTools(Map<String, Object> body) {
        Object choice = body.get("tool_choice");
        if (choice != null && !choice.equals("auto") && !choice.equals("none"))
            throw new IllegalArgumentException("only tool_choice 'auto' and 'none' are supported yet");
        if ("none".equals(choice) || !(body.get("tools") instanceof List<?> list)) return List.of();
        var tools = new ArrayList<ToolSpec>();
        for (Object o : list) {
            Map<String, Object> tool = (Map<String, Object>) o;
            if (!"function".equals(Json.str(tool, "type", "function")))
                throw new IllegalArgumentException("only function tools are supported");
            Map<String, Object> fn = (Map<String, Object>) tool.get("function");
            if (fn == null) throw new IllegalArgumentException("a tool needs a function object");
            String name = Json.str(fn, "name", null);
            if (name == null || name.isBlank()) throw new IllegalArgumentException("a tool needs a function name");
            Object parameters = fn.get("parameters");
            tools.add(new ToolSpec(name, Json.str(fn, "description", ""),
                    parameters == null ? "{}" : Json.write(parameters)));
        }
        return tools;
    }

    /** One facade tool call in OpenAI wire shape; {@code index} is set only for streaming deltas. */
    static Map<String, Object> toolCallJson(ChatContent.ToolCall call, Integer index) {
        Map<String, Object> json = new LinkedHashMap<>();
        if (index != null) json.put("index", index);
        json.put("id", call.id());
        json.put("type", "function");
        json.put("function", Map.of("name", call.name(), "arguments", call.argumentsJson()));
        return json;
    }

    private static ChatRole chatRole(String role) {
        return switch (role) {
            case "system" -> ChatRole.SYSTEM;
            case "user" -> ChatRole.USER;
            case "assistant" -> ChatRole.ASSISTANT;
            case "tool" -> ChatRole.TOOL;
            default -> throw new IllegalArgumentException("unknown role: " + role);
        };
    }

    // ── Non-streaming ─────────────────────────────────────────────────────────

    private void fullResponse(
            HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat,
            LogprobRequest logprobs)
            throws IOException {
        // n choices are generated sequentially on the serialized session, choice i sampling
        // with seed + i so the choices can actually differ; at temperature 0 they are the same
        // completion n times, as greedy choices are. Prompt tokens are counted once in usage,
        // completion tokens across every choice, matching OpenAI's accounting.
        List<Object> choices = new ArrayList<>(logprobs.n());
        int promptTokens = 0;
        int completionTokens = 0;
        try {
            for (int index = 0; index < logprobs.n(); index++) {
                InferenceService.Result r =
                        service.generate(index == 0 ? req : req.withSeed(req.seed() + index), null);
                promptTokens = r.promptTokens();
                completionTokens += r.completionTokens();
                String finish = !r.toolCalls().isEmpty() ? "tool_calls" : r.stopped() ? "stop" : "length";
                Map<String, Object> choice = new LinkedHashMap<>();
                choice.put("index", index);
                if (chat) {
                    // Only chat carries reasoning_content; a text completion's single text field stays
                    // exactly as generated.
                    ReasoningSplit split = splitReasoning(r.text());
                    Map<String, Object> message = new LinkedHashMap<>();
                    message.put("role", "assistant");
                    message.put(
                            "content", r.text().isEmpty() && !r.toolCalls().isEmpty() ? null : split.content());
                    if (split.reasoningContent() != null)
                        message.put("reasoning_content", split.reasoningContent());
                    if (!r.toolCalls().isEmpty())
                        message.put("tool_calls", r.toolCalls().stream().map(call -> toolCallJson(call, null)).toList());
                    choice.put("message", message);
                } else {
                    choice.put("text", r.text());
                }
                if (logprobs.enabled()) {
                    choice.put("logprobs", logprobsPayload(r.tokenLogprobs()));
                }
                choice.put("finish_reason", finish);
                choices.add(choice);
            }
        } catch (IllegalArgumentException e) {
            // The facade rejects a per-request reasoning override the loaded model's format
            // cannot express with exactly this type; that is the client's error, not the server's.
            sendError(ex, 400, e.getMessage());
            return;
        } catch (Exception e) {
            sendError(ex, 500, "Generation failed: " + e);
            return;
        }

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", chat ? "chat.completion" : "text_completion");
        resp.put("created", created);
        resp.put("model", servedModel);
        resp.put("choices", choices);
        resp.put(
                "usage",
                Map.of(
                        "prompt_tokens", promptTokens,
                        "completion_tokens", completionTokens,
                        "total_tokens", promptTokens + completionTokens));
        sendJson(ex, 200, resp);
    }

    // ── Streaming (Server-Sent Events) ────────────────────────────────────────

    private void streamResponse(
            HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat,
            LogprobRequest logprobs)
            throws IOException {
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        ex.getResponseHeaders().set("Connection", "keep-alive");
        ex.sendResponseHeaders(200, 0);
        OutputStream os = ex.getResponseBody();

        String object = chat ? "chat.completion.chunk" : "text_completion";
        try {
            if (chat) {
                // First chunk carries the assistant role.
                writeSse(os, chunk(id, object, created, roleDelta(), null));
            }
            // With tools in play the facade parses calls out of the complete text, so raw
            // token pieces would leak tool markup as content; buffer and emit structured
            // deltas once generation finishes instead.
            boolean buffered = chat && !req.tools().isEmpty();
            // Chat pieces are classified as they arrive: reasoning_content while inside the
            // think block, content after it closes. Text completions keep their single field.
            ReasoningStreamParser reasoningParser = chat && !buffered ? new ReasoningStreamParser() : null;
            InferenceService.Result r;
            if (buffered) {
                r = service.generate(req, null);
            } else if (logprobs.enabled()) {
                // Per-token events, so each chunk can carry its token's logprob entry. The entry
                // rides the first chunk its token's text produces; a token whose text the
                // reasoning parser still holds (a split <think> tag, a partial character) gets
                // an empty-delta chunk, so concatenating every chunk's logprobs.content yields
                // exactly one entry per generated token, in order.
                r = service.generate(
                        req,
                        null,
                        event -> {
                            try {
                                Map<String, Object> entry =
                                        event.tokenLogprob() == null
                                                ? null
                                                : logprobsPayload(List.of(event.tokenLogprob()));
                                if (reasoningParser != null) {
                                    var deltas = reasoningParser.feed(event.text());
                                    if (deltas.isEmpty()) {
                                        writeSse(os, chunk(id, object, created, Map.of(), null, entry));
                                    }
                                    for (int i = 0; i < deltas.size(); i++) {
                                        var delta = deltas.get(i);
                                        writeSse(
                                                os,
                                                chunk(
                                                        id,
                                                        object,
                                                        created,
                                                        delta.reasoning()
                                                                ? reasoningDelta(delta.text())
                                                                : contentDelta(delta.text()),
                                                        null,
                                                        i == 0 ? entry : null));
                                    }
                                } else {
                                    writeSse(
                                            os,
                                            chunk(
                                                    id,
                                                    object,
                                                    created,
                                                    textField(event.text()),
                                                    null,
                                                    entry));
                                }
                            } catch (IOException io) {
                                throw new RuntimeException(io); // client disconnected — abort
                            }
                        });
            } else {
                r = service.generate(
                        req,
                        piece -> {
                            try {
                                if (reasoningParser != null) {
                                    for (ReasoningStreamParser.Delta delta :
                                            reasoningParser.feed(piece)) {
                                        writeSse(
                                                os,
                                                chunk(
                                                        id,
                                                        object,
                                                        created,
                                                        delta.reasoning()
                                                                ? reasoningDelta(delta.text())
                                                                : contentDelta(delta.text()),
                                                        null));
                                    }
                                } else {
                                    writeSse(
                                            os,
                                            chunk(
                                                    id,
                                                    object,
                                                    created,
                                                    textField(piece),
                                                    null));
                                }
                            } catch (IOException io) {
                                throw new RuntimeException(
                                        io); // client disconnected — abort generation
                            }
                        });
            }
            if (reasoningParser != null) {
                for (ReasoningStreamParser.Delta delta : reasoningParser.finish()) {
                    writeSse(
                            os,
                            chunk(
                                    id,
                                    object,
                                    created,
                                    delta.reasoning()
                                            ? reasoningDelta(delta.text())
                                            : contentDelta(delta.text()),
                                    null));
                }
            }
            if (buffered) {
                ReasoningSplit split = splitReasoning(r.text());
                if (split.reasoningContent() != null && !split.reasoningContent().isEmpty())
                    writeSse(os, chunk(id, object, created, reasoningDelta(split.reasoningContent()), null));
                if (!split.content().isEmpty())
                    writeSse(os, chunk(id, object, created, contentDelta(split.content()), null));
                if (!r.toolCalls().isEmpty()) {
                    var calls = new ArrayList<Map<String, Object>>();
                    for (int i = 0; i < r.toolCalls().size(); i++) calls.add(toolCallJson(r.toolCalls().get(i), i));
                    writeSse(os, chunk(id, object, created, Map.of("tool_calls", calls), null));
                }
                if (logprobs.enabled() && !r.tokenLogprobs().isEmpty()) {
                    // Buffered generation produced no per-token chunks to ride: one chunk
                    // carries every entry, which clients concatenate the same way.
                    writeSse(os, chunk(id, object, created, Map.of(), null,
                            logprobsPayload(r.tokenLogprobs())));
                }
            }
            String finish = !r.toolCalls().isEmpty() ? "tool_calls" : r.stopped() ? "stop" : "length";
            writeSse(os, chunk(id, object, created, chat ? Map.of() : textField(""), finish));
            os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            os.flush();
        } catch (Exception e) {
            // Best-effort: nothing more we can send on a half-written SSE stream.
        } finally {
            os.close();
        }
    }

    private Map<String, Object> roleDelta() {
        Map<String, Object> d = new LinkedHashMap<>();
        d.put("role", "assistant");
        d.put("content", "");
        return d;
    }

    private Map<String, Object> contentDelta(String piece) {
        return Map.of("content", piece);
    }

    private Map<String, Object> reasoningDelta(String piece) {
        return Map.of("reasoning_content", piece);
    }

    private Map<String, Object> textField(String piece) {
        return Map.of("text", piece);
    }

    /** One SSE data object. {@code delta} is the chat delta map or the completion text map. */
    private String chunk(
            String id, String object, long created, Map<String, Object> delta, String finish) {
        return chunk(id, object, created, delta, finish, null);
    }

    /** As {@link #chunk(String, String, long, Map, String)}, with a {@code logprobs} object. */
    private String chunk(
            String id, String object, long created, Map<String, Object> delta, String finish,
            Map<String, Object> logprobs) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        if (object.startsWith("chat")) {
            choice.put("delta", delta);
        } else {
            choice.putAll(delta); // {"text": ...}
        }
        if (logprobs != null) {
            choice.put("logprobs", logprobs);
        }
        choice.put("finish_reason", finish);
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("id", id);
        obj.put("object", object);
        obj.put("created", created);
        obj.put("model", servedModel);
        obj.put("choices", List.of(choice));
        return Json.write(obj);
    }

    private void writeSse(OutputStream os, String jsonData) throws IOException {
        os.write(("data: " + jsonData + "\n\n").getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    // ── Wire helpers ──────────────────────────────────────────────────────────

    private void sendJson(HttpExchange ex, int status, Map<String, Object> body)
            throws IOException {
        byte[] bytes = Json.write(body).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    /**
     * {@code POST /v1/systemone}: a state plus typed questions, answered with probabilities. See
     * {@link SystemOneService}.
     */
    private void handleSystemOne(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed — use POST");
            return;
        }
        if (systemOne == null) {
            sendError(
                    ex,
                    501,
                    "/v1/systemone is not available in this serving mode"
                            + " (it needs the per-request session path, not --batch)");
            return;
        }
        Map<String, Object> body;
        try {
            String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            body = Json.parseObject(raw);
        } catch (Exception e) {
            sendError(ex, 400, "Invalid JSON body: " + e.getMessage());
            return;
        }
        try {
            sendJson(ex, 200, systemOne.evaluate(body));
        } catch (SystemOneService.BadRequest bad) {
            sendError(ex, bad.status, bad.getMessage());
        } catch (RuntimeException failure) {
            sendError(ex, 500, "decision failed: " + failure.getMessage());
        }
    }

    /** {@code options} with room for one more concurrent session. */
    static ModelOptions withExtraSession(ModelOptions options) {
        ModelOptions.Builder b =
                ModelOptions.builder()
                        .contextLength(options.contextLength())
                        .maxConcurrentSessions(options.maxConcurrentSessions() + 1)
                        .thinkingMode(options.thinkingMode())
                        .reasoningEffort(options.reasoningEffort());
        if (options.executionPolicy() != null) b.executionPolicy(options.executionPolicy());
        // Copying the resolved storage unconditionally would mark the default as an
        // explicit choice and defeat the device model's own storage default (FP8 here).
        if (options.hasExplicitStorageOptions()) b.storageOptions(options.storageOptions());
        if (options.backend() != null) b.backend(options.backend());
        if (options.device() != null) b.device(options.device());
        return b.build();
    }

    private void sendError(HttpExchange ex, int status, String message) throws IOException {
        Map<String, Object> err =
                Map.of("error", Map.of("message", message, "type", "invalid_request_error"));
        sendJson(ex, status, err);
    }
}

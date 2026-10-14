package org.beehive.jitllm.server;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.FinishReason;
import org.beehive.jitllm.api.GenerationEvent;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.ChatContent;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.ReasoningEffort;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.api.ThinkingMode;
import org.beehive.jitllm.api.TokenLogprob;
import org.beehive.jitllm.api.ToolSpec;

/**
 * Reusable, thread-safe inference wrapper over one loaded model.
 *
 * <p>Holds a single {@link GenerationSession} and serializes generation on it — the GPU is a single
 * context, so requests run one at a time. The session owns the KV lease, the execution plan, the
 * chat template and the incremental decode, which is why none of that appears here.
 *
 * <p>Each request is independent: the server's endpoints carry the whole conversation in {@code
 * messages}, so the session is {@link GenerationSession#reset() reset} before every generation
 * rather than accumulating history of its own.
 */
public final class InferenceService {

    private final LocalModel model;
    private final GenerationSession session;
    private final Object lock = new Object();

    public InferenceService(LocalModel model) {
        this.model = model;
        this.session = ((TextGenerationModel) model).newSession();
    }

    public org.beehive.jitllm.runtime.backend.ExecutionInfo prepare() {
        synchronized (lock) {
            return session.prepare();
        }
    }

    public LocalModel model() {
        return model;
    }

    /**
     * A single generation request. {@code messages} are role/content turns (chat); a lone user turn
     * covers the completions endpoint.
     */
    public record Request(
            List<ChatMessage> messages, int maxTokens, float temperature, float topP, long seed,
            int topK, float minP, float frequencyPenalty, float presencePenalty, float repetitionPenalty,
            List<ToolSpec> tools,
            ThinkingMode thinkingMode, ReasoningEffort reasoningEffort,
            boolean logprobs, int topLogprobs) {
        public Request {
            tools = tools == null ? List.of() : List.copyOf(tools);
        }
        public Request(List<ChatMessage> messages, int maxTokens, float temperature, float topP, long seed,
                int topK, float minP, float frequencyPenalty, float presencePenalty, float repetitionPenalty,
                List<ToolSpec> tools) {
            this(messages, maxTokens, temperature, topP, seed, topK, minP, frequencyPenalty, presencePenalty, repetitionPenalty, tools, null, null, false, 0);
        }
        public Request(List<ChatMessage> messages, int maxTokens, float temperature, float topP, long seed,
                int topK, float minP, float frequencyPenalty, float presencePenalty, float repetitionPenalty) {
            this(messages, maxTokens, temperature, topP, seed, topK, minP, frequencyPenalty, presencePenalty, repetitionPenalty, List.of());
        }
        public Request(List<ChatMessage> messages, int maxTokens, float temperature, float topP, long seed) {
            this(messages, maxTokens, temperature, topP, seed, 0, 0, 0, 0, 1, List.of());
        }

        /** This request with a different sampling seed — one choice of an {@code n > 1} request. */
        Request withSeed(long newSeed) {
            return new Request(messages, maxTokens, temperature, topP, newSeed, topK, minP,
                    frequencyPenalty, presencePenalty, repetitionPenalty, tools,
                    thinkingMode, reasoningEffort, logprobs, topLogprobs);
        }
    }

    /** Result of a generation: the text, any tool calls, and the token counts (for {@code usage}). */
    public record Result(String text, int promptTokens, int completionTokens, boolean stopped,
            List<ChatContent.ToolCall> toolCalls, List<TokenLogprob> tokenLogprobs) {
        public Result {
            toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
            tokenLogprobs = tokenLogprobs == null ? List.of() : List.copyOf(tokenLogprobs);
        }
        public Result(String text, int promptTokens, int completionTokens, boolean stopped,
                List<ChatContent.ToolCall> toolCalls) {
            this(text, promptTokens, completionTokens, stopped, toolCalls, List.of());
        }
        public Result(String text, int promptTokens, int completionTokens, boolean stopped) {
            this(text, promptTokens, completionTokens, stopped, List.of(), List.of());
        }
    }

    /**
     * Generate a completion. If {@code onToken} is non-null each decoded token's text is streamed
     * to it as it is produced; the full text is always returned. Thread-safe (serialized).
     */
    public Result generate(Request req, java.util.function.Consumer<String> onToken) {
        return generate(req, onToken, null);
    }

    /**
     * As {@link #generate(Request, java.util.function.Consumer)}, additionally delivering each
     * {@link GenerationEvent} — token id, text and, when the request asked for them,
     * log-probabilities — to {@code onEvent} as it is produced, for streaming responses that
     * carry per-token logprobs.
     */
    public Result generate(
            Request req,
            java.util.function.Consumer<String> onToken,
            java.util.function.Consumer<GenerationEvent> onEvent) {
        // Counted here, around the one call that produces a GenerationResult, because its
        // prompt/generated token counts are the authoritative ones /metrics reports; the HTTP
        // layer only ever estimated prompt size (4 chars per token) for its window guard.
        Metrics.beginGeneration();
        long durationNanos = 0;
        int promptTokens = 0;
        int completionTokens = 0;
        try {
        synchronized (lock) {
            long startedNanos = System.nanoTime();
            try {
            session.reset();
            GenerationRequest.Builder builder =
                    GenerationRequest.builder()
                            .messages(req.messages())
                            .maxNewTokens(req.maxTokens() > 0 ? req.maxTokens() : 256)
                            .temperature(req.temperature())
                            .topP(req.topP())
                            .seed(req.seed())
                            .topK(req.topK())
                            .minP(req.minP())
                            .frequencyPenalty(req.frequencyPenalty())
                            .presencePenalty(req.presencePenalty())
                            .repetitionPenalty(req.repetitionPenalty())
                            .tools(req.tools())
                            .thinkingMode(req.thinkingMode())
                            .reasoningEffort(req.reasoningEffort())
                            .logprobs(req.logprobs())
                            .topLogprobs(req.topLogprobs());
            if (onToken != null || onEvent != null) {
                builder.onEvent(
                        event -> {
                            if (onEvent != null) {
                                onEvent.accept(event);
                            }
                            if (onToken != null && !event.text().isEmpty()) {
                                onToken.accept(event.text());
                            }
                        });
            }
            GenerationResult result = session.generate(builder.build());
            promptTokens = result.promptTokens();
            completionTokens = result.generatedTokens();
            return new Result(
                    result.text(),
                    result.promptTokens(),
                    result.generatedTokens(),
                    result.finishReason() == FinishReason.STOP_TOKEN,
                    result.toolCalls(),
                    result.tokenLogprobs());
            } finally {
                durationNanos = System.nanoTime() - startedNanos;
            }
        }
        } finally {
            Metrics.finishGeneration(durationNanos, promptTokens, completionTokens);
        }
    }

    /**
     * Process-wide counters behind {@code GET /metrics} (Prometheus text exposition).
     *
     * <p>Generation numbers are recorded in {@link #generate} above, from the {@link
     * GenerationResult} itself, so counted tokens are the tokens the facade actually processed —
     * never the HTTP layer's 4-chars-per-token estimate. The engine-backed serving path bypasses
     * this class entirely, so its adapter in {@code OpenAIServer} records through the same methods
     * using its {@link Result} counts; either way a generation is counted exactly once. Request
     * counts by endpoint are recorded by the HTTP handlers, the only place that knows the endpoint.
     * State is static because the exposition is process-wide (the same choice the codebase already
     * makes for {@code RunMetrics}).
     */
    public static final class Metrics {

        private static final Map<String, AtomicLong> REQUESTS_BY_ENDPOINT =
                new ConcurrentHashMap<>();
        private static final AtomicLong GENERATIONS = new AtomicLong();
        private static final AtomicLong PROMPT_TOKENS = new AtomicLong();
        private static final AtomicLong COMPLETION_TOKENS = new AtomicLong();
        private static final AtomicLong DURATION_NANOS = new AtomicLong();
        private static final AtomicLong RUNNING = new AtomicLong();

        private Metrics() {}

        /** One HTTP request was handled on {@code endpoint} (a path, never user input). */
        public static void recordRequest(String endpoint) {
            REQUESTS_BY_ENDPOINT.computeIfAbsent(endpoint, key -> new AtomicLong())
                    .incrementAndGet();
        }

        /**
         * A generation started. Counted before the session lock is taken, so the gauge covers
         * callers queued on the serialized session as well as the one executing.
         */
        public static void beginGeneration() {
            RUNNING.incrementAndGet();
        }

        /**
         * A generation finished, successfully or not. Token counts are added as given: a failed
         * generation reports none and so adds none.
         */
        public static void finishGeneration(
                long durationNanos, int promptTokens, int completionTokens) {
            RUNNING.decrementAndGet();
            GENERATIONS.incrementAndGet();
            PROMPT_TOKENS.addAndGet(promptTokens);
            COMPLETION_TOKENS.addAndGet(completionTokens);
            DURATION_NANOS.addAndGet(durationNanos);
        }

        /** The Prometheus text exposition (version 0.0.4) of every counter above. */
        public static String render() {
            StringBuilder out = new StringBuilder();
            out.append("# HELP jitllm_requests_total HTTP requests handled, by endpoint.\n");
            out.append("# TYPE jitllm_requests_total counter\n");
            for (Map.Entry<String, AtomicLong> entry :
                    new TreeMap<>(REQUESTS_BY_ENDPOINT).entrySet()) {
                out.append("jitllm_requests_total{endpoint=\"")
                        .append(escapeLabel(entry.getKey()))
                        .append("\"} ")
                        .append(entry.getValue().get())
                        .append('\n');
            }
            out.append("# HELP jitllm_generations_total Generations finished, including failed"
                    + " ones.\n");
            out.append("# TYPE jitllm_generations_total counter\n");
            out.append("jitllm_generations_total ").append(GENERATIONS.get()).append('\n');
            out.append("# HELP jitllm_prompt_tokens_total Prompt tokens reported by finished"
                    + " generations.\n");
            out.append("# TYPE jitllm_prompt_tokens_total counter\n");
            out.append("jitllm_prompt_tokens_total ").append(PROMPT_TOKENS.get()).append('\n');
            out.append("# HELP jitllm_completion_tokens_total Completion tokens reported by"
                    + " finished generations.\n");
            out.append("# TYPE jitllm_completion_tokens_total counter\n");
            out.append("jitllm_completion_tokens_total ")
                    .append(COMPLETION_TOKENS.get())
                    .append('\n');
            out.append("# HELP jitllm_generation_duration_seconds Generation call duration in"
                    + " seconds.\n");
            out.append("# TYPE jitllm_generation_duration_seconds summary\n");
            out.append("jitllm_generation_duration_seconds_sum ")
                    .append(Double.toString(DURATION_NANOS.get() / 1_000_000_000.0))
                    .append('\n');
            out.append("jitllm_generation_duration_seconds_count ")
                    .append(GENERATIONS.get())
                    .append('\n');
            out.append("# HELP jitllm_running_requests Generations in flight, including callers"
                    + " queued on the serialized session.\n");
            out.append("# TYPE jitllm_running_requests gauge\n");
            out.append("jitllm_running_requests ").append(RUNNING.get()).append('\n');
            return out.toString();
        }

        private static String escapeLabel(String value) {
            return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
        }

        /** Test-only reset; production counters run for the life of the process. */
        static void reset() {
            REQUESTS_BY_ENDPOINT.clear();
            GENERATIONS.set(0);
            PROMPT_TOKENS.set(0);
            COMPLETION_TOKENS.set(0);
            DURATION_NANOS.set(0);
            RUNNING.set(0);
        }
    }

    /** Close the session; the model outlives it and is closed by the caller that loaded it. */
    public void close() {
        synchronized (lock) {
            session.close();
        }
    }
}

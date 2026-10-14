package org.beehive.jitllm.api;

/**
 * One piece of a conversation turn.
 *
 * <p><b>Sealed</b>, so a new kind is a reviewed decision rather than something that arrives
 * because an interface was open. {@link Image} is that decision, taken for the Flash-Next
 * vision tower: it carries encoded image bytes, and families without a vision path reject it
 * at encode time rather than silently dropping it.
 *
 * <p>The JSON-bearing strings are <b>opaque</b>. This milestone introduces no JSON-schema model:
 * both integrations already hold JSON at this boundary, the engine only splices it into a prompt,
 * and a modelled schema would be a second, worse copy of a specification this project does not own.
 */
public sealed interface ChatContent {

    /** Plain text. */
    record Text(String text) implements ChatContent {
        public Text {
            if (text == null) {
                throw new IllegalArgumentException(
                        "message text must not be null;"
                                + " use an empty string if a turn genuinely has no text");
            }
        }
    }

    /**
     * A tool the assistant asked to run.
     *
     * @param id the call's identifier, non-blank — it is what a {@link ToolResult} is matched
     *     against
     * @param name the tool's name, non-blank
     * @param argumentsJson the arguments as an opaque JSON string, never {@code null}
     */
    record ToolCall(String id, String name, String argumentsJson) implements ChatContent {
        public ToolCall {
            requireNonBlank(id, "a tool call id");
            requireNonBlank(name, "a tool name");
            if (argumentsJson == null) {
                throw new IllegalArgumentException(
                        "tool call arguments must not be null;"
                                + " an argumentless call is an empty JSON object, not null");
            }
        }
    }

    /**
     * What running a tool produced.
     *
     * @param id the {@link ToolCall#id()} this answers, non-blank
     * @param name the tool's name, non-blank
     * @param resultJson the result as an opaque JSON string, never {@code null}
     */
    record ToolResult(String id, String name, String resultJson) implements ChatContent {
        public ToolResult {
            requireNonBlank(id, "a tool result id");
            requireNonBlank(name, "a tool name");
            if (resultJson == null) {
                throw new IllegalArgumentException(
                        "a tool result must not be null;"
                                + " a tool that returns nothing returns an empty JSON object, not null");
            }
        }
    }

    /**
     * One encoded image (PNG, JPEG, or another format the model's image preprocessor decodes).
     *
     * <p>Immutable: the bytes are copied on construction and again on every access, so neither
     * the caller's array nor an array handed out earlier can change what a built request carries.
     * The size cap is a decode-safety bound, not a model limit — it rejects payloads that no
     * image the processor would accept needs, before any decoder sees them.
     *
     * @param encodedBytes the encoded image, non-empty and at most {@value #MAX_ENCODED_BYTES}
     *     bytes (32 MiB)
     */
    record Image(byte[] encodedBytes) implements ChatContent {

        /** Largest encoded image accepted: 32 MiB. */
        public static final int MAX_ENCODED_BYTES = 32 * 1024 * 1024;

        public Image {
            if (encodedBytes == null || encodedBytes.length == 0) {
                throw new IllegalArgumentException(
                        "image bytes must not be null or empty; an image piece carries an encoded"
                                + " image or is not sent at all");
            }
            if (encodedBytes.length > MAX_ENCODED_BYTES) {
                throw new IllegalArgumentException(
                        "encoded image is "
                                + encodedBytes.length
                                + " bytes, over the "
                                + MAX_ENCODED_BYTES
                                + "-byte (32 MiB) limit");
            }
            encodedBytes = encodedBytes.clone();
        }

        @Override
        public byte[] encodedBytes() {
            return encodedBytes.clone();
        }
    }

    private static void requireNonBlank(String value, String what) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    what
                            + " must not be blank, got "
                            + (value == null ? "null" : "'" + value + "'"));
        }
    }
}

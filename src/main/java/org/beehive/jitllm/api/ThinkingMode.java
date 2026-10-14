package org.beehive.jitllm.api;

/**
 * Whether a model that has a reasoning phase should use it.
 *
 * <p>It exists because a real consumer needs it. The Quarkus extension publishes {@code
 * enable-thinking} as configuration and implemented it by reaching into {@code ChatFormat};
 * migrating it onto the facade without this would have silently dropped a published property.
 *
 * <h2>Where it lives, and where it does not</h2>
 *
 * <p>On {@link ModelOptions} as a model default and on {@link SessionOptions} as a per-session
 * override, resolved once when the session opens — the same shape as execution policy. {@code
 * GenerationRequest} may additionally override it for a single request ({@code null} there means
 * "the session's mode"): the HTTP serving surface needs that, because OpenAI clients carry
 * {@code chat_template_kwargs.enable_thinking} per request against one long-lived session, and
 * the only thing a mode changes is how that request's prompt is encoded. Resolution is request
 * over session over model, and an unset request override leaves prior behaviour bit-identical.
 */
public enum ThinkingMode {

    /**
     * Whatever the family does on its own.
     *
     * <p>Nothing is encoded, so a family with no reasoning phase is unaffected and one that has a
     * default keeps it. This is the value that changes nothing, and it is the default.
     */
    DEFAULT,

    /** Ask for the reasoning phase. Rejected by a model that cannot represent the control. */
    ENABLED,

    /**
     * Ask the model to skip the reasoning phase. Rejected by a model that cannot represent the
     * control.
     *
     * <p>Rejected rather than ignored: a caller who turned thinking off and silently got it anyway
     * pays for tokens they asked not to generate, and nothing tells them.
     */
    DISABLED;

    /** Whether this mode asks the model for anything, as opposed to leaving it alone. */
    public boolean isExplicit() {
        return this != DEFAULT;
    }
}

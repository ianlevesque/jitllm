package org.beehive.jitllm.inference.state;

/** Request-owned Flash-Next recurrent, sparse-index and PLE history. */
public interface Qwen4SequenceState extends AutoCloseable {
    int position();
    void reset();

    /**
     * Stages an image-bearing prompt plan. Only the target engine sequence consumes one;
     * draft and test sequences refuse rather than silently run image tokens through the
     * text embedding table.
     */
    default void stageMultimodalPrompt(MultimodalPromptPlan plan) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " cannot ingest image content");
    }

    @Override void close() throws java.io.IOException;
}

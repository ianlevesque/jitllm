package org.beehive.jitllm.inference.state;

/** Request-owned Flash-Next recurrent, sparse-index and PLE history. */
public interface Qwen4SequenceState extends AutoCloseable {
    int position();
    void reset();
    @Override void close() throws java.io.IOException;
}

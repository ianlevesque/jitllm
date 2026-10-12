package org.beehive.jitllm.inference.state;

/** Request-owned Flash-Next recurrent, sparse-index and PLE history. */
public interface Qwen4SequenceState {
    int position();
    void reset();
}

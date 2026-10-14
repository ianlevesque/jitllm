package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/** Host adapter; the sequence owns lazy KV allocation rather than the legacy dense cache. */
public final class Qwen4State extends State implements AutoCloseable, MultimodalState {
    private final Qwen4SequenceState sequence;

    public Qwen4State(Configuration config, int batchsize, Qwen4SequenceState sequence) {
        super(config, batchsize);
        this.sequence = java.util.Objects.requireNonNull(sequence);
    }

    public Qwen4SequenceState sequence() { return sequence; }

    @Override public void stageMultimodalPrompt(MultimodalPromptPlan plan) {
        sequence.stageMultimodalPrompt(plan);
    }

    @Override protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        fields.xb = fields.xb2 = fields.hb = fields.hb2 = fields.q = fields.k = fields.v = fields.att = ArrayFloatTensor.allocate(1);
        fields.keyCache = fields.valueCache = new FloatTensor[0];
        return fields;
    }

    @Override public void resetSequenceState() { sequence.reset(); }
    @Override public void close() throws java.io.IOException { sequence.close(); }
    @Override public void setPosition(int position) {
        if (position != sequence.position()) throw new IllegalArgumentException("Sequence position mismatch");
    }
    @Override public void resetPositionHolder() {}
    @Override public State clone() throws CloneNotSupportedException {
        throw new CloneNotSupportedException("Flash-Next state needs an explicit sequence snapshot; shallow cloning shares recurrent state");
    }
}

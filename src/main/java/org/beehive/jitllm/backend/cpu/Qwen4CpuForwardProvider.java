package org.beehive.jitllm.backend.cpu;

import java.io.UncheckedIOException;
import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/** Service-loaded native Flash-Next host forward pass. */
public final class Qwen4CpuForwardProvider implements CpuForwardProvider {
    @Override public ArchitectureId architecture() { return Qwen4CpuModel.ARCHITECTURE; }
    @Override public ForwardPass create() {
        return (model, state, token, position) -> {
            var nativeModel = (Qwen4CpuModel) model;
            var nativeState = (Qwen4State) state;
            var sequence = (Qwen4CpuEngine.Sequence) nativeState.sequence();
            if (position != sequence.position()) throw new IllegalArgumentException("Flash-Next forward position " + position + " does not match sequence " + sequence.position());
            try {
                var output = nativeModel.engine().forward(token, sequence, sequence.requiresLogits(), null);
                for (int i = 0; i < output.hidden().length; i++) state.x.setFloat(i, output.hidden()[i]);
                if(output.logits()!=null) for (int i = 0; i < output.logits().length; i++) state.logits.setFloat(i, output.logits()[i]);
            } catch (java.io.IOException failure) { throw new UncheckedIOException(failure); }
        };
    }
}

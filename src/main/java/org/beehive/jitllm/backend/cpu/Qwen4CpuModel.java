package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.Qwen4ChatFormat;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;

/** Normal JITLLM model adapter for the native host forward pass. */
public class Qwen4CpuModel extends AbstractModel implements AutoCloseable {
    static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen4_exp");
    private final Qwen4CpuEngine engine;

    private record MixedWeights(Qwen4CpuEngine engine) implements Weights {
        // The legacy single-type API reports the dense projection type. Expert matrices
        // retain their own NVFP4 descriptors and are never materialized as BF16 weights.
        @Override public DataType dataType() { return DataType.BF16; }
    }

    public Qwen4CpuModel(Qwen4CpuEngine engine) {
        super(engine.tokenizer(), new MixedWeights(engine), new Qwen4ChatFormat(engine.tokenizer()));
        this.engine = engine;
    }
    public Qwen4CpuEngine engine() { return engine; }
    @Override public Qwen4Configuration configuration() { return engine.configuration(); }
    @Override public Qwen4Tokenizer tokenizer() { return engine.tokenizer(); }
    @Override public ModelType getModelType() { return ModelType.QWEN_4_EXP; }
    @Override public ArchitectureId architectureId() { return ARCHITECTURE; }
    @Override public boolean shouldAddBeginOfText() { return false; }
    @Override public int kvCacheDim() { return configuration().numberOfKeyValueHeads() * configuration().headSize(); }
    @Override public State createNewState() { return createNewState(-1); }
    @Override public State createNewState(int batchsize) {
        return State.withPrefillBatchSize(1, () -> {
            var state = new Qwen4State(configuration(), batchsize, engine.newSequence());
            // This loop ingests every prompt token; there is no separate BOS seed.
            // A matching header seed would make the API under-count the prompt by one.
            state.latestToken = -1;
            return state;
        });
    }
    @Override public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated) {
        var sequence=(Qwen4CpuEngine.Sequence)((Qwen4State)state).sequence();
        // The last sampled token is emitted before it is fed back. API positions include
        // it, so ingest that pending token before appending a new turn's prompt.
        try {
            if(sequence.position()+1==startPosition && state.latestToken>=0) engine.forward(state.latestToken,sequence,false,null);
            if(sequence.position()!=startPosition) throw new IllegalArgumentException("Flash-Next generation position mismatch");
            if(!promptTokens.isEmpty()) sequence.beginPrefill(promptTokens.size());
            var generated=TokenGenerationLoop.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens,
                    maxTokens, sampler, echo, onTokenGenerated);
            if(!generated.isEmpty()) state.latestToken=generated.getLast();
            return generated;
        } catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}
        finally {sequence.finishPrefill();}
    }
    @Override public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated,
            TornadoVMMasterPlan plan) {
        throw new UnsupportedOperationException("Flash-Next full GPU execution is not implemented yet");
    }
    @Override public void close() throws IOException { engine.close(); }
}

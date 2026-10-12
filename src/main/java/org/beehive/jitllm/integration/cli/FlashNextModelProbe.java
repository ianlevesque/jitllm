package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.backend.cpu.CpuForwardPasses;
import org.beehive.jitllm.backend.cpu.Qwen4CpuModel;
import org.beehive.jitllm.server.Json;

/** Exercise the registered source/provider/forward path on the original checkpoint. */
public final class FlashNextModelProbe {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("Usage: FlashNextModelProbe MODEL_DIR REQUEST.json");
        var request = Json.parseObject(Files.readString(Path.of(args[1])));
        long start = System.nanoTime();
        try (var model = (Qwen4CpuModel) ModelLoader.loadModel(Path.of(args[0]), -1, true, false)) {
            var state = model.createNewState();
            var forward = CpuForwardPasses.forArchitecture(model.architectureId());
            var input = ((List<?>) request.get("input_tokens")).stream().map(v -> ((Number)v).intValue()).toList();
            if (input.isEmpty()) throw new IllegalArgumentException("Nonempty input required");
            ((org.beehive.jitllm.backend.cpu.Qwen4CpuEngine.Sequence)((org.beehive.jitllm.inference.state.Qwen4State)state).sequence()).beginInitialPrefill(input.size());
            for (int i = 0; i < input.size(); i++) {
                System.err.println("Provider forward " + (i + 1) + "/" + input.size());
                forward.forward(model, state, input.get(i), i);
            }
            int best = 0;
            for (int i = 1; i < state.logits.size(); i++) if (state.logits.getFloat(i) > state.logits.getFloat(best)) best = i;
            var ordered = new Integer[model.tokenizer().tokenCount()];
            for (int i = 0; i < ordered.length; i++) ordered[i] = i;
            Arrays.sort(ordered, Comparator.comparingDouble((Integer i) -> -state.logits.getFloat(i)));
            var top = new ArrayList<Object>();
            for (int i = 0; i < 10; i++) top.add(Map.of("token", ordered[i], "logit", (double)state.logits.getFloat(ordered[i])));
            var result = new LinkedHashMap<String,Object>();
            result.put("architecture", model.architectureId().toString());
            result.put("input_tokens", input); result.put("next_token", best);
            result.put("text", model.tokenizer().decode(List.of(best)));
            result.put("top_logits", top);
            result.put("cache_resident_bytes", model.engine().statistics().residentBytes());
            result.put("elapsed_seconds", (System.nanoTime() - start) / 1e9);
            System.out.println(Json.write(result));
        }
    }
}

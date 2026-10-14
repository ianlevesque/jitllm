package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.Map;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.Qwen4PleTable;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.provider.ModelProvider;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/** Recognize Flash-Next without opening tensor payloads. */
public final class Qwen4ModelProvider implements ModelProvider {
    @Override public boolean supports(ModelSource source) {
        Object type = source.metadata().get("model_type");
        return "qwen4_exp_text".equals(type) || ("qwen4_exp".equals(type)
                && source.metadata().get("text_config") instanceof Map<?,?> text
                && "qwen4_exp_text".equals(text.get("model_type")));
    }
    @Override public ArchitectureId architecture(ModelSource source) { return Qwen4CpuModel.ARCHITECTURE; }
    @Override public Model load(ModelSource source, BackendId backend, int contextLength) throws IOException {
        if (!supports(source) || !source.isSafetensors()) throw new IOException("Expected a Qwen4-Exp safetensors directory: " + source.path());
        long budget = Long.getLong("jitllm.qwen4.pageCacheBytes", 1048576L);
        // Engine/GPU load signatures carry only the dense budget, so forward the resolved PLE
        // budget (default 256 MiB, 0 = legacy shared cache) through its own property for
        // Qwen4CpuLayers.Ple to pick up when it constructs each layer's table.
        System.setProperty(Qwen4PleTable.PLE_PAGE_CACHE_PROPERTY, Long.toString(Qwen4PleTable.plePageCacheBytes()));
        if (BackendId.CUDA.equals(backend)) return org.beehive.jitllm.backend.tornado.Qwen4GpuModel.load(source.path(), contextLength, budget);
        if (!BackendId.CPU.equals(backend)) throw new UnsupportedOperationException("Flash-Next supports CPU or CUDA, not " + backend);
        return new Qwen4CpuModel(new Qwen4CpuEngine(source.path(), contextLength, budget, null));
    }
}

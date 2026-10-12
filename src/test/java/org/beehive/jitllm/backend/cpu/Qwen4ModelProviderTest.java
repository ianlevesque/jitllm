package org.beehive.jitllm.backend.cpu;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.Map;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.provider.ModelProviders;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;
import static org.junit.Assert.*;

public class Qwen4ModelProviderTest {
    @Test public void servicesSelectTheExactArchitecture() {
        var source = ModelSource.ofMetadata(Path.of("catalogue"), Map.of("model_type", "qwen4_exp",
                "text_config", Map.of("model_type", "qwen4_exp_text")));
        assertTrue(ModelProviders.select(source) instanceof Qwen4ModelProvider);
        assertEquals(Qwen4CpuModel.ARCHITECTURE, ModelProviders.select(source).architecture(source));
        assertNotNull(CpuForwardPasses.forArchitecture(Qwen4CpuModel.ARCHITECTURE));
    }
    @Test public void otherArchitecturesAreNotClaimed() {
        var provider = new Qwen4ModelProvider();
        assertFalse(provider.supports(ModelSource.ofMetadata(Path.of("other"), Map.of("model_type", "qwen3_5"))));
        assertFalse(provider.supports(ModelSource.ofMetadata(Path.of("other"), Map.of("model_type", "qwen4_exp"))));
    }
    @Test public void gpuLoadIsRefusedBeforeAnyTensorRead() throws Exception {
        Path directory = Files.createTempDirectory("qwen4-provider-");
        try {
            Files.writeString(directory.resolve("config.json"), "{\"model_type\":\"qwen4_exp_text\"}");
            Files.writeString(directory.resolve("model.safetensors"), "unread payload");
            var source = ModelSource.ofFile(directory);
            var failure = assertThrows(UnsupportedOperationException.class,
                    () -> new Qwen4ModelProvider().load(source, BackendId.CUDA, -1));
            assertTrue(failure.getMessage().contains("full GPU execution"));
        } finally {
            Files.deleteIfExists(directory.resolve("config.json"));
            Files.deleteIfExists(directory.resolve("model.safetensors"));
            Files.delete(directory);
        }
    }
}

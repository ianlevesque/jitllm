package org.beehive.jitllm.format;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;

public class SafetensorsModelSourceTest {
    private static ModelSource source(String config) throws Exception {
        Path directory = Files.createTempDirectory("safetensors-source-");
        try {
            Files.writeString(directory.resolve("config.json"), config);
            // Invalid tensor bytes deliberately: recognition must never parse payloads.
            Files.writeString(directory.resolve("model.safetensors"), "unread payload");
            return ModelSource.ofFile(directory);
        } finally {
            Files.deleteIfExists(directory.resolve("config.json"));
            Files.deleteIfExists(directory.resolve("model.safetensors"));
            Files.delete(directory);
        }
    }
    @Test public void metadataRecognitionDoesNotReadTensors() throws Exception {
        ModelSource source = source("{\"model_type\":\"qwen4_exp\",\"wide\":9007199254740993}");
        assertTrue(source.isSafetensors()); assertTrue(source.isLoadable());
        assertEquals("qwen4_exp", source.metadataString("model_type"));
        assertEquals(9007199254740993L, source.metadata().get("wide"));
        assertThrows(IllegalStateException.class, source::gguf);
        assertThrows(UnsupportedOperationException.class, () -> source.metadata().put("model_type", "other"));
    }
    @Test public void malformedConfigIsAnIoFailure() {
        assertThrows(IOException.class, () -> source("{\"model_type\":}"));
    }
    @Test public void directoryWithoutWeightsFailsRecognition() throws Exception {
        Path directory = Files.createTempDirectory("missing-weights-");
        try { assertThrows(IOException.class, () -> ModelSource.ofFile(directory)); }
        finally { Files.delete(directory); }
    }
}

package org.beehive.jitllm.server;

import static org.junit.Assert.*;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;

public final class OpenAIServerOptionsTest {
    @Test public void extraSessionKeepsDefaultStorageImplicit() {
        var options = ModelOptions.builder().contextLength(4096).maxConcurrentSessions(1).build();
        assertFalse(options.hasExplicitStorageOptions());
        var widened = OpenAIServer.withExtraSession(options);
        assertEquals(2, widened.maxConcurrentSessions());
        assertFalse("resolved default storage must not become explicit", widened.hasExplicitStorageOptions());
    }
    @Test public void extraSessionPreservesExplicitStorage() {
        var options = ModelOptions.builder().contextLength(4096).maxConcurrentSessions(3)
                .storageOptions(StorageOptions.fp8()).build();
        var widened = OpenAIServer.withExtraSession(options);
        assertEquals(4, widened.maxConcurrentSessions());
        assertTrue(widened.hasExplicitStorageOptions());
        assertEquals(StorageOptions.fp8(), widened.storageOptions());
    }
}

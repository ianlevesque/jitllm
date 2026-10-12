package org.beehive.jitllm.integration.cli;

import java.nio.file.Path;
import org.beehive.jitllm.server.Json;

public final class FlashNextNvfp4Probe {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) throw new IllegalArgumentException("Usage: FlashNextNvfp4Probe MODEL_DIR MATRIX_PREFIX DEVICE_INDEX");
        System.out.println(Json.write(org.beehive.jitllm.backend.tornado.FlashNextNvfp4Probe.run(
                Path.of(args[0]), args[1], Integer.parseInt(args[2]))));
    }
}

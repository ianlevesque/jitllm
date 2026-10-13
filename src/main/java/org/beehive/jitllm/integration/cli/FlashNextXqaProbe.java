package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import org.beehive.jitllm.backend.tornado.FlashNextXqaNativeProbe;
import org.beehive.jitllm.server.Json;

/** Portable entry point for the native XQA conformance probe. */
public final class FlashNextXqaProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=1) throw new IllegalArgumentException("Usage: FlashNextXqaProbe REQUEST.json");
        System.out.println(Json.write(FlashNextXqaNativeProbe.run(Json.parseObject(Files.readString(Path.of(args[0]))))));
    }
}

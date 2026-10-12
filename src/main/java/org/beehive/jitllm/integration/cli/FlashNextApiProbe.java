package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.api.*;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.format.safetensors.MetadataJson;

/** Real-checkpoint API loading, three-session admission, reset and resource release. */
public final class FlashNextApiProbe {
    private static long descriptors(Set<String> checkpointFiles) throws Exception {
        try(var files=Files.list(Path.of("/proc/self/fd"))) {
            return files.filter(file -> {
                try {return checkpointFiles.contains(Files.readSymbolicLink(file).toString());}
                catch(Exception closed) {return false;}
            }).count();
        }
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=1) throw new IllegalArgumentException("Usage: FlashNextApiProbe MODEL_DIR");
        Path checkpoint=Path.of(args[0]).toRealPath();
        Set<String> checkpointFiles=new HashSet<>();
        try(var files=Files.list(checkpoint)) {
            for(var file:files.filter(path->path.toString().endsWith(".safetensors")).toList()) checkpointFiles.add(file.toRealPath().toString());
        }
        long before=descriptors(checkpointFiles);
        var options=ModelOptions.builder().backend(BackendId.CPU).maxConcurrentSessions(3).build();
        var model=LocalModels.load(checkpoint,options);
        var generator=(TextGenerationModel)model;
        var sessions=new ArrayList<GenerationSession>();
        long during;
        String architecture=model.info().architecture();
        try {
            for(int i=0;i<3;i++) {
                var session=generator.newSession();sessions.add(session);
                session.prepare();session.reset();
                if(session.position()!=0) throw new AssertionError("Reset advanced API session");
            }
            try {generator.newSession();throw new AssertionError("Fourth session admitted");}
            catch(IllegalStateException expected) {}
            try {model.close();throw new AssertionError("Model closed with live sessions");}
            catch(IllegalStateException expected) {}
            during=descriptors(checkpointFiles);
        } finally {for(var session:sessions) session.close();model.close();}
        model.close();
        long after=descriptors(checkpointFiles);
        if(after!=before) throw new AssertionError("Checkpoint file descriptors leaked: "+before+" -> "+after);
        System.out.println(MetadataJson.write(Map.of("architecture",architecture,"context_length",model.info().contextLength(),
                "sessions",3,"fourth_session_refused",true,"close_with_live_sessions_refused",true,"reset_exact",true,
                "checkpoint_descriptors_before",before,"checkpoint_descriptors_during",during,"checkpoint_descriptors_after",after)));
    }
}

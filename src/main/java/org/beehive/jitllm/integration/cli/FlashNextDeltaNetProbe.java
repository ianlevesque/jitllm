package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.Qwen4DeltaNet;
import org.beehive.jitllm.backend.tornado.FlashNextDeltaNetExecutor;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.server.Json;

/** Actual CUDA recurrence, nested device rollback, bounded checkpoints and sequence release. */
public final class FlashNextDeltaNetProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("Usage: FlashNextDeltaNetProbe MODEL_DIR REQUEST.json");
        var model=Path.of(args[0]);var request=Json.parseObject(Files.readString(Path.of(args[1])));
        var config=Qwen4Configuration.load(model,-1);String prefix="model.language_model.layers.0.linear_attn";
        var index=SafeTensorIndex.open(model,name->name.startsWith(prefix+"."));
        int device=((Number)request.getOrDefault("device",1)).intValue();
        boolean projected=Boolean.TRUE.equals(request.get("gpu_projections"));
        try(var reader=new PagedTensorReader(1048576,4096);
            var matrices=projected ? new FlashNextMatrixExecutor(device,256L<<20) : null;
            var executor=new FlashNextDeltaNetExecutor(device,4,256L<<20)) {
            var layer=new Qwen4DeltaNet(config,index,reader,prefix,matrices,executor);
            try(var tiny=new FlashNextDeltaNetExecutor(device,4,1)) {
                var refused=new Qwen4DeltaNet(config,index,reader,prefix,null,tiny);
                try {refused.newState();throw new AssertionError("GPU state exceeded its budget");}
                catch(IllegalStateException expected) {}
                if(tiny.statistics().residentBytes()!=0) throw new AssertionError("Rejected state retained memory");
            }
            var outputs=new ArrayList<Object>();
            float[] first=values((List<?>)((List<?>)request.get("inputs")).getFirst());
            try(var state=layer.newState();var foreign=layer.newState()) {
                for(Object value:(List<?>)request.get("inputs")) {
                    float[] input=values((List<?>)value);
                    try(var before=state.snapshot()) {
                        float[] output=layer.forward(input,state);
                        try(var accepted=state.snapshot()) {
                            state.restore(before);
                            if(!Arrays.equals(output,layer.forward(input,state))) throw new AssertionError("GPU DeltaNet replay mismatch");
                            state.restore(accepted);
                        }
                        outputs.add(floats(output));
                    }
                    if(executor.statistics().liveSnapshots()!=0) throw new AssertionError("GPU checkpoint leaked");
                }
                try(var saved=state.snapshot()) {
                    var forged=new Qwen4DeltaNet.State.Snapshot(foreign,new float[0],new float[0],saved.device());
                    try {foreign.restore(forged);throw new AssertionError("Foreign checkpoint accepted");}
                    catch(IllegalArgumentException expected) {}
                }
                var snapshots=new ArrayList<Qwen4DeltaNet.State.Snapshot>();
                try {
                    for(int i=0;i<4;i++) snapshots.add(state.snapshot());
                    try {state.snapshot();throw new AssertionError("Unbounded GPU checkpoint pool");}
                    catch(IllegalStateException expected) {}
                } finally {for(var saved:snapshots) saved.close();}
                var stale=state.snapshot();state.reset();
                try(var current=state.snapshot()) {
                    try {state.restore(stale);throw new AssertionError("Checkpoint survived reset");}
                    catch(IllegalArgumentException expected) {}
                    stale.close();
                    if(executor.statistics().liveSnapshots()!=1) throw new AssertionError("Stale close released a current checkpoint");
                    try {state.restore(stale);throw new AssertionError("Released checkpoint accepted");}
                    catch(IllegalArgumentException expected) {}
                    state.restore(current);
                }
                if(!floats(layer.forward(first,state)).equals(outputs.getFirst())) throw new AssertionError("GPU reset mismatch");
                if(!floats(layer.forward(first,foreign)).equals(outputs.getFirst())) throw new AssertionError("GPU sequences share recurrence");
            }
            var stats=executor.statistics();
            if(stats.residentBytes()!=0 || stats.sessions()!=0 || stats.liveSnapshots()!=0) throw new AssertionError("GPU state leaked after close");
            var report=new LinkedHashMap<String,Object>();
            report.put("outputs",outputs);report.put("rollback_exact",true);report.put("reset_exact",true);
            report.put("checkpoint_pool_bounded",true);report.put("foreign_checkpoint_refused",true);
            report.put("state_budget_refused",true);
            report.put("released_checkpoint_refused",true);report.put("independent_sequences",true);
            report.put("resident_bytes_after_close",stats.residentBytes());report.put("state_host_download_bytes",stats.stateHostDownloadBytes());
            report.put("gpu_forwards",stats.forwards());report.put("gpu_checkpoints",stats.checkpoints());report.put("gpu_restores",stats.restores());
            report.put("gpu_resets",stats.resets());report.put("gpu_kernel_ns",stats.kernelNanos());
            report.put("gpu_projections",projected);report.put("cache_resident_bytes",reader.statistics().residentBytes());
            System.out.println(Json.write(report));
        }
    }
    private static float[] values(List<?> values) {
        float[] input=new float[values.size()];for(int i=0;i<input.length;i++) input[i]=FloatFormats.roundBf16(((Number)values.get(i)).floatValue());return input;
    }
    private static List<Double> floats(float[] values) {var list=new ArrayList<Double>();for(float value:values) list.add((double)value);return list;}
}

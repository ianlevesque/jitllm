package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** XQA native boundary, stream/caller changes and same-JVM library reload checks. */
public final class FlashNextXqaNativeProbe {
    private static BFloat16Array load(Arena arena,String file,int elements) throws Exception {
        byte[] raw=Files.readAllBytes(Path.of(file));if(raw.length!=2*elements) throw new IllegalArgumentException("Wrong XQA fixture bytes");
        var result=BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+raw.length,32));
        result.getSegment().copyFrom(java.lang.foreign.MemorySegment.ofArray(raw));return result;
    }
    public static Map<String,Object> run(Map<String,Object> request) throws Exception {
        int device=((Number)request.getOrDefault("device",1)).intValue();
        int capacity=2112,heads=24,kvHeads=2,dimension=256;var results=new ArrayList<Object>();long kernel=0;
        for(int reload=0;reload<2;reload++) {
            try(var arena=Arena.ofShared()) {
                var q=load(arena,(String)request.get("query"),heads*dimension);
                var k=load(arena,(String)request.get("keys"),capacity*kvHeads*dimension);var v=load(arena,(String)request.get("values"),k.getSize());
                var output=BFloat16Array.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+2L*q.getSize(),32));
                var host=FloatArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+4L*q.getSize(),32));
                var control=new IntArray(6);var pages=new IntArray(capacity/64);for(int i=0;i<pages.getSize();i++) pages.set(i,i);
                var graph=new TaskGraph("flashnextXqaProbe"+reload).transferToDevice(DataTransferMode.FIRST_EXECUTION,q,k,v,output,pages)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,control)
                        .libraryTask("decode",FlashNextXqaLibrary::decode,capacity,dimension,heads,kvHeads,control,q,k,v,output,pages)
                        .task("read",FlashNextWeightKernels::bf16GemmOutput,output,host).transferToHost(DataTransferMode.EVERY_EXECUTION,host);
                try(var plan=new TornadoExecutionPlan(graph.snapshot()).withDevice(TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(device)).withProfiler(ProfilerMode.SILENT)) {
                    for(var count:(List<?>)request.get("counts")) {
                        control.set(2,((Number)count).intValue());kernel+=plan.execute().getProfilerResult().getDeviceKernelTime();float[] first=host.toHeapArray();
                        kernel+=plan.execute().getProfilerResult().getDeviceKernelTime();if(!Arrays.equals(first,host.toHeapArray())) throw new AssertionError("XQA replay differs");
                        var task=new FutureTask<float[]>(()-> {plan.execute();return host.toHeapArray();});Thread.ofPlatform().start(task);
                        if(!Arrays.equals(first,task.get())) throw new AssertionError("XQA caller stream differs");
                        var row=new ArrayList<Double>();for(float value:first) row.add((double)value);
                        results.add(Map.of("reload",reload,"selected_tokens",count,"output",row,"replay_exact",true,"caller_thread_exact",true));
                    }
                }
            }
            if(FlashNextXqaLibrary.statistics().contexts()!=0 || FlashNextXqaLibrary.statistics().workspaceBytes()!=0) throw new AssertionError("XQA resources survived close");
        }
        var stats=FlashNextXqaLibrary.statistics();if(kernel<=0 || stats.calls()==0) throw new AssertionError("No native XQA execution");
        return Map.of("passed",true,"cases",results,"native_calls",stats.calls(),"kernel_ns",kernel,"workspace_bytes_per_context",FlashNextXqaLibrary.WORKSPACE_BYTES,"resources_after_close",0);
    }
}

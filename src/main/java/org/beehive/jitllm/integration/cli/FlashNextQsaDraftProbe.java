package org.beehive.jitllm.integration.cli;

import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextQsaExecutor;
import org.beehive.jitllm.inference.op.QsaExecutor;
import org.beehive.jitllm.server.Json;

/** Sparse-budget and page-boundary qualification for three frozen-index MTP steps. */
public final class FlashNextQsaDraftProbe {
    private static void constantAttention(float[] output) {for(float value:output) if(value!=.5f) throw new AssertionError("Constant-V sparse attention differs: "+value);}
    private static int[] selection(int[] captured,int length,int tail) {int[] out=Arrays.copyOf(captured,captured.length+tail);for(int i=0;i<tail;i++) out[captured.length+i]=length+i;return out;}
    public static void main(String[] args) throws Exception {
        int device=args.length==0 ? 1 : Integer.parseInt(args[0]);
        var shape=new QsaExecutor.Shape(4096,2,1,8,2,8,4,2048,4,1e-6f,3);
        float[] norms=new float[8],frequencies={1,.01f};int[] axes={0,0};
        float[] query=new float[32],key=new float[8],value=new float[8],index=new float[24];
        Arrays.fill(key,1);Arrays.fill(value,1);for(int head=0;head<2;head++) Arrays.fill(query,head*16,head*16+8,1);
        try(var factory=new FlashNextQsaExecutor(device,8,4L<<20);var state=factory.newSession(shape,norms,norms,norms,norms,axes,frequencies,true,true)) {
            for(int token=0;token<2063;token++) {
                for(int i=0;i<index.length;i++) index[i]=(float)Math.sin(token*.017+i*.23);
                constantAttention(state.forward(query,key,value,index,new int[]{token,token,token},false,false,32));
            }
            int capturedLength=state.tokens();int[] captured=state.selectedTokens();
            if(captured.length!=2051 || shape.selectedCapacity()!=2054 || new HashSet<Integer>(Arrays.stream(captured).boxed().toList()).size()!=captured.length)
                throw new AssertionError("Sparse MTP capture geometry differs");
            long downloads=factory.statistics().indexHostDownloadBytes();
            try(var base=state.snapshot()) {
                for(int step=0;step<3;step++) {
                    var selected=selection(captured,capturedLength,step+1);int position=state.tokens();
                    constantAttention(state.forwardFrozen(query,key,value,new int[]{position,position,position},selected));
                    if(!Arrays.equals(selected,state.selectedTokens())) throw new AssertionError("MTP frozen selection changed");
                }
                if(factory.statistics().indexHostDownloadBytes()!=downloads) throw new AssertionError("MTP draft downloaded indexer results");
                int position=state.tokens();
                try {state.forwardFrozen(query,key,value,new int[]{position,position,position},selection(captured,capturedLength,4));throw new AssertionError("Oversized MTP tail accepted");} catch(IllegalArgumentException expected) {}
                try {state.forward(query,key,value,index,new int[]{position,position,position},false,false,32);throw new AssertionError("Unrestored draft prefix accepted");} catch(IllegalArgumentException expected) {}
                if(state.tokens()!=position) throw new AssertionError("Refused draft changed position");
                state.restore(base);
                for(int step=0;step<3;step++) {int pos=state.tokens();constantAttention(state.forwardFrozen(query,key,value,new int[]{pos,pos,pos},selection(captured,capturedLength,step+1)));}
                state.restore(base);
                int[] invalid=selection(captured,capturedLength,1);invalid[invalid.length-1]=state.tokens()+1;
                try {state.forwardFrozen(query,key,value,new int[]{capturedLength,capturedLength,capturedLength},invalid);throw new AssertionError("Future sparse token accepted");} catch(IllegalArgumentException expected) {}
                try(var other=factory.newSession(shape,norms,norms,norms,norms,axes,frequencies,true,true)) {try {other.restore(base);throw new AssertionError("Foreign sparse checkpoint accepted");} catch(IllegalArgumentException expected) {}}
            }
            var expired=state.snapshot();expired.close();try {state.restore(expired);throw new AssertionError("Disposed sparse checkpoint accepted");} catch(IllegalArgumentException expected) {}
            var stale=state.snapshot();state.reset();try {state.restore(stale);throw new AssertionError("Stale sparse checkpoint accepted");} catch(IllegalArgumentException expected) {}stale.close();
            Arrays.fill(index,0);constantAttention(state.forward(query,key,value,index,new int[]{0,0,0},false,false,32));
            var stats=factory.statistics();state.close();factory.close();
            if(factory.statistics().reservedBytes()!=0 || factory.statistics().sessions()!=0 || factory.statistics().liveSnapshots()!=0 || stats.kvHostDownloadBytes()!=0)
                throw new AssertionError("MTP sparse resources leaked");
            var report=new LinkedHashMap<String,Object>();report.put("passed",true);report.put("captured_length",capturedLength);report.put("captured_selection",captured.length);
            report.put("selected_capacity",shape.selectedCapacity());report.put("draft_steps",3);report.put("rollback_replay_exact",true);report.put("draft_index_download_bytes",0);
            report.put("pages_reached",stats.pages());report.put("kv_download_bytes",stats.kvHostDownloadBytes());report.put("kernel_ns",stats.kernelNanos());report.put("resources_after_close",0);
            System.out.println(Json.write(report));
        }
    }
}

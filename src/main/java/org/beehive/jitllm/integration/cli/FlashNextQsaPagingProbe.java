package org.beehive.jitllm.integration.cli;

import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextQsaExecutor;
import org.beehive.jitllm.format.safetensors.FloatFormats;
import org.beehive.jitllm.format.safetensors.Nvfp4Activation;
import org.beehive.jitllm.inference.op.QsaExecutor;
import org.beehive.jitllm.server.Json;

/** Execute page transitions and bounded top-512 selection with an independent host oracle.
 * Reduced head widths isolate page/index semantics; this is not full-model long-context validation.
 */
public final class FlashNextQsaPagingProbe {
    private static final int DIM=8,HEADS=4,KV_HEADS=2,RATIO=4,BUDGET=2048;
    private static float bf(float value) {return FloatFormats.roundBf16(value);}
    private static float fp8(float value) {
        return FloatFormats.e4m3(((Float.floatToRawIntBits(value)>>>24)&128)|Nvfp4Activation.e4m3(Math.abs(value)));
    }
    private static float[] normalize(float[] input,int width) {
        float[] out=new float[input.length];
        for(int start=0;start<input.length;start+=width) {
            double sum=0;for(int i=0;i<width;i++) sum+=input[start+i]*input[start+i];
            float factor=(float)(1/Math.sqrt((float)(sum/width)+1e-6f));
            for(int i=0;i<width;i++) out[start+i]=bf(input[start+i]*factor);
        }
        return out;
    }
    private record Input(float[] qg,float[] k,float[] v,float[] index) {}
    private static Input input(int token) {
        float[] qg=new float[2*HEADS*DIM],key=new float[KV_HEADS*DIM],value=new float[key.length],index=new float[12];
        for(int head=0;head<HEADS;head++) {qg[head*2*DIM]=1;qg[head*2*DIM+1]=.5f;qg[head*2*DIM+2]=-.25f;}
        for(int i=0;i<key.length;i++) {
            key[i]=bf((float)Math.sin(token*.013+i*.7));
            value[i]=bf((float)Math.sin(token*.009+i*.3)*.75f);
        }
        index[0]=1;index[4]=1;index[8]=1;index[9]=bf(.125f+(token/RATIO%257)/32f);
        return new Input(qg,key,value,index);
    }
    private static QsaExecutor.Session session(FlashNextQsaExecutor executor,int context) {
        return executor.newSession(new QsaExecutor.Shape(context,HEADS,KV_HEADS,DIM,2,4,RATIO,BUDGET,0,1e-6f),
                new float[DIM],new float[DIM],new float[4],new float[4],new int[0],new float[0],true,true);
    }
    private static float[] forward(QsaExecutor.Session session,int token,boolean fresh) throws Exception {
        var x=input(token);return session.forward(x.qg,x.k,x.v,x.index,new int[]{token,token/2,token%7},fresh,fresh,32);
    }
    private static int[] selection(int visible) {
        var blocks=new ArrayList<Integer>();for(int i=0;i<visible/RATIO;i++) blocks.add(i);
        blocks.sort((left,right)-> {
            float l=normalize(Arrays.copyOfRange(input(left*RATIO).index,8,12),4)[0];
            float r=normalize(Arrays.copyOfRange(input(right*RATIO).index,8,12),4)[0];
            int order=Float.compare(r,l);return order!=0 ? order : Integer.compare(left,right);
        });
        int count=Math.min(blocks.size(),BUDGET/RATIO)*RATIO+visible%RATIO;int[] out=new int[count];
        for(int i=0;i<count;i++) out[i]=i<count-visible%RATIO ? blocks.get(i/RATIO)*RATIO+i%RATIO : visible-visible%RATIO+i-(count-visible%RATIO);
        return out;
    }
    private static float[] reference(int token,int[] selected,boolean fresh) {
        var raw=input(token);float[] query=new float[HEADS*DIM];
        for(int h=0;h<HEADS;h++) System.arraycopy(raw.qg,h*2*DIM,query,h*DIM,DIM);
        query=normalize(query,DIM);float[][] keys=new float[selected.length][],values=new float[selected.length][];
        for(int row=0;row<selected.length;row++) {
            var x=input(selected[row]);keys[row]=normalize(x.k,DIM);values[row]=x.v.clone();
            if(!fresh) for(int i=0;i<keys[row].length;i++) {keys[row][i]=fp8(keys[row][i]);values[row][i]=fp8(values[row][i]);}
        }
        float[] out=new float[HEADS*DIM];
        for(int h=0;h<HEADS;h++) {
            int kh=h/(HEADS/KV_HEADS);float[] scores=new float[selected.length];
            for(int row=0;row<scores.length;row++) {
                float dot=0;for(int i=0;i<DIM;i++) {
                    float q=query[h*DIM+i];if(fresh) q=bf(q/(float)Math.sqrt(DIM)*1.4426950408f);
                    dot=Math.fma(q,keys[row][kh*DIM+i],dot);
                }
                scores[row]=fresh ? dot : dot/(float)Math.sqrt(DIM);
            }
            if(fresh) {
                float maximum=Float.NEGATIVE_INFINITY,denominator=0;float[] sums=new float[DIM];
                for(int start=0;start<scores.length;start+=32) {
                    int end=Math.min(start+32,scores.length);float next=maximum;
                    for(int row=start;row<end;row++) next=Math.max(next,scores[row]);
                    float alpha=(float)Math.pow(2,maximum-next),norm=0;
                    for(int i=0;i<DIM;i++) sums[i]*=alpha;
                    for(int row=start;row<end;row++) {
                        float p=(float)Math.pow(2,scores[row]-next);norm+=p;p=bf(p);
                        for(int i=0;i<DIM;i++) sums[i]=Math.fma(p,values[row][kh*DIM+i],sums[i]);
                    }
                    denominator=Math.fma(denominator,alpha,norm);maximum=next;
                }
                for(int i=0;i<DIM;i++) out[h*DIM+i]=bf(bf(sums[i]/denominator)*.5f);
            } else {
                float maximum=Float.NEGATIVE_INFINITY;for(float score:scores) maximum=Math.max(maximum,score);
                double denominator=0;float[] p=new float[scores.length];
                for(int row=0;row<scores.length;row++) {p[row]=(float)Math.exp(scores[row]-maximum);denominator+=p[row];}
                for(int i=0;i<DIM;i++) {
                    float sum=0;for(int row=0;row<scores.length;row++) sum=Math.fma((float)(p[row]/denominator),values[row][kh*DIM+i],sum);
                    out[h*DIM+i]=bf(bf(sum)*.5f);
                }
            }
        }
        return out;
    }
    private static void rejects(Class<? extends Throwable> expected,Checked action) throws Exception {
        try {action.run();} catch(Throwable failure) {if(expected.isInstance(failure)) return;throw failure;}
        throw new AssertionError("Expected "+expected.getSimpleName());
    }
    private interface Checked {void run() throws Exception;}
    public static void main(String[] args) throws Exception {
        int device=args.length==0 ? 1 : Integer.parseInt(args[0]);long start=System.nanoTime();
        var cases=new ArrayList<Object>();long reservation,materialized;int pageCount;
        try(var insufficient=new FlashNextQsaExecutor(device,4,1)) {
            rejects(IllegalStateException.class,()->session(insufficient,32));
            if(insufficient.statistics().reservedBytes()!=0 || insufficient.statistics().sessions()!=0) throw new AssertionError("Refused allocation retained reservation");
        }
        try(var executor=new FlashNextQsaExecutor(device,4,64L<<20);var state=session(executor,262144);var other=session(executor,32)) {
            reservation=executor.statistics().reservedBytes();
            if(executor.statistics().pages()!=0) throw new AssertionError("Unused pages materialized");
            var closed=other.snapshot();closed.close();rejects(IllegalArgumentException.class,()->other.restore(closed));
            try(var saved=state.snapshot()) {rejects(IllegalArgumentException.class,()->other.restore(saved));}
            try(var a=state.snapshot();var b=state.snapshot();var c=state.snapshot();var d=state.snapshot()) {rejects(IllegalStateException.class,state::snapshot);}
            try(var stale=state.snapshot()) {state.reset();rejects(IllegalArgumentException.class,()->state.restore(stale));}
            float[] first=null;
            Set<Integer> checks=Set.of(0,3,2047,2048,2051,4095,4096,4098);
            for(int token=0;token<4099;token++) {
                boolean fresh=token<4096;
                try(var saved=checks.contains(token) ? state.snapshot() : null) {
                    float[] observed=forward(state,token,fresh);if(token==0) first=observed.clone();
                    if(checks.contains(token)) {
                        int[] selected=state.selectedTokens(),expected=selection(token+1);
                        if(!Arrays.equals(selected,expected)) throw new AssertionError("Selection differs at token "+token);
                        float[] ref=reference(token,expected,fresh);float error=0;
                        for(int i=0;i<ref.length;i++) {error=Math.max(error,Math.abs(ref[i]-observed[i]));if(Math.abs(ref[i]-observed[i])>.004f+.016f*Math.abs(ref[i])) throw new AssertionError("Attention differs at token "+token+" column "+i);}
                        if(fresh) state.finishFreshPrefill();
                        try(var future=state.snapshot()) {
                            state.restore(saved);rejects(IllegalArgumentException.class,()->state.restore(future));
                            if(!Arrays.equals(observed,forward(state,token,fresh)) || !Arrays.equals(selected,state.selectedTokens())) throw new AssertionError("Page-boundary replay differs");
                        }
                        cases.add(Map.of("token",token,"fresh_prefill",fresh,"selected_tokens",selected.length,"max_absolute_error",(double)error,"rollback_exact",true));
                        System.err.println("Checked QSA page boundary token "+token);
                    }
                }
            }
            materialized=executor.statistics().materializedPageBytes();pageCount=executor.statistics().pages();
            if(pageCount!=3 || materialized>=reservation || executor.statistics().kvHostDownloadBytes()!=0 || executor.statistics().liveSnapshots()!=0) throw new AssertionError("Invalid page residency/checkpoints");
            if(!Arrays.equals(first,forward(other,0,true))) throw new AssertionError("Independent sequence differs");
            for(int token=1;token<32;token++) forward(other,token,true);
            rejects(IllegalArgumentException.class,()->forward(other,32,true));
            other.close();state.reset();if(!Arrays.equals(first,forward(state,0,true))) throw new AssertionError("Reset differs");
            state.close();rejects(IllegalStateException.class,state::snapshot);rejects(IllegalStateException.class,()->forward(state,0,true));
            if(executor.statistics().reservedBytes()!=0 || executor.statistics().sessions()!=0 || executor.statistics().pages()!=0) throw new AssertionError("QSA close leaked buffers");
            var stats=executor.statistics();
            System.out.println(Json.write(Map.of("passed",true,"configured_context",262144,"executed_tokens",4099,"page_tokens",2048,"materialized_pages",pageCount,
                    "reserved_bytes",reservation,"materialized_page_bytes",materialized,"cases",cases,"gpu",Map.of("kernel_ns",stats.kernelNanos(),"forwards",stats.forwards(),"restores",stats.restores(),"kv_host_download_bytes",stats.kvHostDownloadBytes(),"reserved_bytes_after_close",stats.reservedBytes()),
                    "elapsed_seconds",(System.nanoTime()-start)/1e9)));
        }
    }
}

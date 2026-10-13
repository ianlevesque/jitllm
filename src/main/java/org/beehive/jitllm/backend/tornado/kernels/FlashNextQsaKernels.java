package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Paged KV, compressed-key top-k and sparse GQA reference kernels. */
public final class FlashNextQsaKernels {
    private FlashNextQsaKernels() {}
    private static float bf(float value) {return FlashNextDeltaNetKernels.bf16(value);}
    private static short bits(float value) {return (short)(Float.floatToRawIntBits(bf(value))>>>16);}
    private static float unbits(short value) {return Float.intBitsToFloat((value&65535)<<16);}
    private static void normalizedRope(FloatArray input,int inputStart,FloatArray weights,FloatArray output,int outputStart,
                                       IntArray positions,int positionStart,IntArray axes,FloatArray frequencies,int dimension,float epsilon,int fused) {
        double squares=0;
        for(int i=0;i<dimension;i++) {float value=input.get(inputStart+i);squares+=value*value;}
        float factor=(float)(1/TornadoMath.sqrt((double)((float)(squares/dimension)+epsilon)));
        int half=frequencies.getSize();
        for(int i=0;i<half;i++) {
            float a=input.get(inputStart+i)*factor*(1+weights.get(i));
            float b=input.get(inputStart+half+i)*factor*(1+weights.get(half+i));
            if(fused==0) {a=bf(a);b=bf(b);}
            float angle=positions.get(positionStart+axes.get(i))*frequencies.get(i);
            float cosine=(float)TornadoMath.cos((double)angle),sine=(float)TornadoMath.sin((double)angle);
            if(fused==0) {cosine=bf(cosine);sine=bf(sine);}
            float left=fused==1 ? a*cosine-b*sine : bf(a*cosine)-bf(b*sine);
            float right=fused==1 ? b*cosine+a*sine : bf(b*cosine)+bf(a*sine);
            output.set(outputStart+i,bf(left));output.set(outputStart+half+i,bf(right));
        }
        for(int i=2*half;i<dimension;i++) output.set(outputStart+i,bf(input.get(inputStart+i)*factor*(1+weights.get(i))));
    }
    public static void prepareQkv(FloatArray projected,FloatArray rawKey,FloatArray qNorm,FloatArray kNorm,IntArray positions,
                                   IntArray axes,FloatArray frequencies,FloatArray query,FloatArray key,FloatArray gate,int dimension,float epsilon,int fused) {
        for(@Parallel int head=0;head<query.getSize()/dimension;head++) {
            normalizedRope(projected,head*2*dimension,qNorm,query,head*dimension,positions,0,axes,frequencies,dimension,epsilon,fused);
            for(int i=0;i<dimension;i++) gate.set(head*dimension+i,projected.get(head*2*dimension+dimension+i));
        }
        for(@Parallel int head=0;head<key.getSize()/dimension;head++)
            normalizedRope(rawKey,head*dimension,kNorm,key,head*dimension,positions,0,axes,frequencies,dimension,epsilon,fused);
    }
    public static void prepareIndexer(FloatArray projected,FloatArray pending,IntArray positions,
                                       FloatArray qNorm,IntArray axes,FloatArray frequencies,FloatArray query,
                                       FloatArray pooled,int width,int ratio,float epsilon,int fused,IntArray control) {
        for(@Parallel int head=0;head<query.getSize()/width;head++)
            if(control.get(5)==0)
            normalizedRope(projected,head*width,qNorm,query,head*width,positions,0,axes,frequencies,width,epsilon,fused);
        for(@Parallel int i=0;i<width;i++) {
            if(control.get(5)==0) {
            float sum=0;for(int row=0;row<ratio;row++) sum+=pending.get(row*width+i);
            pooled.set(i,fused==1 ? sum/ratio : bf(sum/ratio));
            }
        }
    }
    public static void normalizePooled(FloatArray pooled,FloatArray weights,FloatArray output,IntArray firstPosition,
                                       IntArray axes,FloatArray frequencies,int width,float epsilon,int fused,IntArray control) {
        // One compressed head. Pooling is a preceding task, so all lanes have completed.
        for(@Parallel int head=0;head<1;head++) if(control.get(5)==0) normalizedRope(pooled,0,weights,output,0,firstPosition,0,axes,frequencies,width,epsilon,fused);
    }
    public static void frozenSelection(IntArray input,IntArray selected) {
        for(@Parallel int i=0;i<selected.getSize();i++) selected.set(i,input.get(i));
    }
    private static int fp8(float value,FloatArray table) {
        int sign=(Float.floatToRawIntBits(value)>>>24)&128;
        float magnitude=TornadoMath.abs(value);int best=0;float distance=Float.MAX_VALUE;
        // Nearest finite E4M3 value, ties to even; clamp at +/-448.
        for(int i=0;i<127;i++) {
            float delta=TornadoMath.abs(magnitude-table.get(i));
            if(delta<distance || (delta==distance && (i&1)==0)) {distance=delta;best=i;}
        }
        return sign|best;
    }
    public static void append(FloatArray key,FloatArray value,FloatArray pooled,FloatArray table,ByteArray cache,
                              BFloat16Array compressed,BFloat16Array fresh,IntArray control,int width,int rows,int indexWidth,int ratio,int fp8) {
        for(@Parallel int i=0;i<2*width;i++) {
            float v=i<width ? key.get(i) : value.get(i-width);int offset=control.get(0)*2*width+i;
            if(fp8==1) cache.set(offset,(byte)fp8(v,table));
            else {short raw=bits(v);cache.set(offset*2,(byte)raw);cache.set(offset*2+1,(byte)(raw>>>8));}
            if(control.get(1)==1) fresh.set(offset,bits(v));
        }
        for(@Parallel int i=0;i<indexWidth;i++)
            if(control.get(2)==1) compressed.set(i*512+control.get(0)/ratio,bits(pooled.get(i)));
    }
    public static void clearTop(FloatArray scores,IntArray indices) {
        for(@Parallel int i=0;i<512;i++) {scores.set(i,Float.NEGATIVE_INFINITY);indices.set(i,Integer.MAX_VALUE);}
    }
    /** Declare producer access to shared inputs/workspaces so named consumers can alias them.
     * Transfer declarations alone do not establish task access in TornadoVM.
     * Every visible gathered entry is overwritten by its owning page before attention.
     */
    public static void initializeShared(FloatArray value,FloatArray table,IntArray control,IntArray selected,BFloat16Array gathered) {
        for(@Parallel int i=0;i<selected.getSize();i++) selected.set(i,i<control.get(2) ? 0 : -1);
        for(@Parallel int i=0;i<gathered.getSize();i++) gathered.set(i,bits(value.get(i%value.getSize())+table.get(0)));
    }
    /** A single 1024-lane workgroup merges 512 new blocks with the previous top 512. */
    public static void mergeTop(KernelContext context,FloatArray query,BFloat16Array compressed,FloatArray bestScore,IntArray bestIndex,
                                IntArray control,int width,int heads) {
        int lane=context.localIdx;
        float[] scores=context.allocateFloatLocalArray(1024);int[] ids=context.allocateIntLocalArray(1024);
        float score=Float.NEGATIVE_INFINITY;int id=Integer.MAX_VALUE;
        if(lane<512) {
            if(lane<control.get(0)) {
                score=0;
                for(int h=0;h<heads;h++) {
                    float dot=0;for(int k=0;k<width;k++) dot+=query.get(h*width+k)*unbits(compressed.get(k*512+lane));
                    score+=TornadoMath.max(dot,0f);
                }
                score/=(float)TornadoMath.sqrt((double)width);id=control.get(1)+lane;
            }
        } else {score=bestScore.get(lane-512);id=bestIndex.get(lane-512);}
        scores[lane]=score;ids[lane]=id;context.localBarrier();
        for(int size=2;size<=1024;size*=2) for(int stride=size/2;stride>0;stride/=2) {
            int peer=lane^stride;
            float a=scores[lane],b=scores[peer];int ai=ids[lane],bi=ids[peer];
            boolean aBetter=a>b || (a==b && ai<bi);
            boolean descending=(lane&size)==0;
            boolean takeBetter=((lane&stride)==0)==descending;
            float result=takeBetter ? (aBetter ? a : b) : (aBetter ? b : a);
            int resultId=takeBetter ? (aBetter ? ai : bi) : (aBetter ? bi : ai);
            context.localBarrier();scores[lane]=result;ids[lane]=resultId;context.localBarrier();
        }
        if(lane<512) {bestScore.set(lane,scores[lane]);bestIndex.set(lane,ids[lane]);}
    }
    public static void selection(IntArray best,IntArray selected,IntArray control,int ratio,int budget) {
        for(@Parallel int i=0;i<selected.getSize();i++) {
            int blocks=TornadoMath.min(control.get(0)/ratio,budget/ratio),tail=control.get(0)%ratio;
            int value=-1;if(i<blocks*ratio) value=best.get(i/ratio)*ratio+i%ratio;
            else if(i<blocks*ratio+tail) value=control.get(0)-tail+i-blocks*ratio;
            selected.set(i,value);
        }
    }
    public static void gather(ByteArray cache,BFloat16Array fresh,FloatArray table,IntArray selected,BFloat16Array gathered,
                                IntArray control,int width,int start,int rows,int fp8) {
        for(@Parallel int i=0;i<gathered.getSize();i++) {
            int selectedRow=i/(2*width),channel=i%(2*width),token=selected.get(selectedRow);
            if(token>=start && token<start+rows) {
                int offset=(token-start)*2*width+channel;short value;
                if(control.get(1)==1) value=fresh.get(offset);
                else if(fp8==1) value=bits(table.get(cache.get(offset)&255));
                else value=(short)((cache.get(offset*2)&255)|((cache.get(offset*2+1)&255)<<8));
                gathered.set(i,value);
            }
        }
    }
    public static void attentionScores(FloatArray query,BFloat16Array gathered,FloatArray scores,IntArray control,int dimension,int kvHeads,int heads,int selectedCapacity,int fused) {
        for(@Parallel int index=0;index<scores.getSize();index++) {
            int head=index/selectedCapacity,row=index%selectedCapacity;
            if(row<control.get(2)) {
                int keyHead=head/(heads/kvHeads);float sum=0;
                for(int k=0;k<dimension;k++) {
                    float q=query.get(head*dimension+k);
                    if(control.get(3)==1 && fused==1) q=bf(q/(float)TornadoMath.sqrt((double)dimension)*1.4426950408f);
                    sum+=q*unbits(gathered.get(row*2*kvHeads*dimension+keyHead*dimension+k));
                }
                scores.set(index,control.get(3)==1 && fused==1 ? sum : sum/(float)TornadoMath.sqrt((double)dimension));
            }
        }
    }
    public static void probabilities(FloatArray scores,FloatArray probabilities,FloatArray alpha,FloatArray denominator,IntArray control,int heads,int capacity,int tileCapacity,int fused) {
        for(@Parallel int head=0;head<heads;head++) {
            int count=control.get(2);
            if(control.get(3)==1 && fused==1) {
                float maximum=Float.NEGATIVE_INFINITY,norm=0;int tile=0;
                for(int start=0;start<count;start+=control.get(4)) {
                    int end=TornadoMath.min(start+control.get(4),count);float next=maximum;
                    for(int row=start;row<end;row++) next=TornadoMath.max(next,scores.get(head*capacity+row));
                    float rescale=(float)TornadoMath.pow(2.0,(double)(maximum-next)),sum=0;alpha.set(head*tileCapacity+tile,rescale);
                    for(int row=start;row<end;row++) {float p=(float)TornadoMath.pow(2.0,(double)(scores.get(head*capacity+row)-next));sum+=p;probabilities.set(head*capacity+row,bf(p));}
                    norm=norm*rescale+sum;maximum=next;tile++;
                }
                denominator.set(head,norm);
            } else {
                float maximum=Float.NEGATIVE_INFINITY;for(int row=0;row<count;row++) maximum=TornadoMath.max(maximum,scores.get(head*capacity+row));
                double norm=0;
                for(int row=0;row<count;row++) {float p=(float)TornadoMath.exp((double)(scores.get(head*capacity+row)-maximum));probabilities.set(head*capacity+row,p);norm+=p;}
                for(int row=0;row<count;row++) probabilities.set(head*capacity+row,(float)(probabilities.get(head*capacity+row)/norm));
                denominator.set(head,1);
            }
        }
    }
    public static void attend(BFloat16Array gathered,FloatArray probabilities,FloatArray alpha,FloatArray denominator,FloatArray gate,FloatArray output,IntArray control,
                                int dimension,int kvHeads,int heads,int capacity,int tileCapacity,int fused) {
        for(@Parallel int index=0;index<output.getSize();index++) {
            int head=index/dimension,column=index%dimension,keyHead=head/(heads/kvHeads),count=control.get(2);
            float sum=0;
            if(control.get(3)==1 && fused==1) {
                int tile=0;
                for(int start=0;start<count;start+=control.get(4)) {
                    sum*=alpha.get(head*tileCapacity+tile);int end=TornadoMath.min(start+control.get(4),count);
                    for(int row=start;row<end;row++) sum+=probabilities.get(head*capacity+row)*unbits(gathered.get(row*2*kvHeads*dimension+kvHeads*dimension+keyHead*dimension+column));
                    tile++;
                }
                sum/=denominator.get(head);
            } else for(int row=0;row<count;row++) sum+=probabilities.get(head*capacity+row)*unbits(gathered.get(row*2*kvHeads*dimension+kvHeads*dimension+keyHead*dimension+column));
            float sigmoid=(float)(1/(1+TornadoMath.exp(-(double)gate.get(index))));
            output.set(index,bf(bf(sum)*(fused==1 ? sigmoid : bf(sigmoid))));
        }
    }
    public static void copySelection(IntArray source,IntArray output) {for(@Parallel int i=0;i<output.getSize();i++) output.set(i,source.get(i));}
}

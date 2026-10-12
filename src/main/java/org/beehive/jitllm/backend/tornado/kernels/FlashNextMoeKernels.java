package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** GPU routing, BF16 SwiGLU and unfused expert/shared finalization. */
public final class FlashNextMoeKernels {
    private FlashNextMoeKernels() {}
    private static float bf(float value) {return FlashNextDeltaNetKernels.bf16(value);}
    private static float value(short raw) {return Float.intBitsToFloat((raw&65535)<<16);}
    public static void bf16Boundary(FloatArray input) {for(@Parallel int i=0;i<input.getSize();i++) input.set(i,bf(input.get(i)));}
    public static void swiglu(FloatArray gate,FloatArray up,FloatArray output) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            float g=gate.get(i);float silu=(float)((double)g/(1+TornadoMath.exp(-(double)g)));
            output.set(i,bf(silu*up.get(i)));
        }
    }
    /** One block selects the top experts stably; CPU receives only the IDs needed to stage weights. */
    public static void route(KernelContext context,FloatArray logits,IntArray selected,FloatArray weights,int normalize) {
        int lane=context.localIdx;float[] scores=context.allocateFloatLocalArray(512);int[] ids=context.allocateIntLocalArray(512);
        scores[lane]=logits.get(lane);ids[lane]=lane;context.localBarrier();
        for(int size=2;size<=512;size*=2) for(int stride=size/2;stride>0;stride/=2) {
            int peer=lane^stride;float a=scores[lane],b=scores[peer];int ai=ids[lane],bi=ids[peer];
            boolean better=a>b || (a==b && ai<bi),descending=(lane&size)==0,takeBetter=((lane&stride)==0)==descending;
            float score=takeBetter ? (better ? a : b) : (better ? b : a);int id=takeBetter ? (better ? ai : bi) : (better ? bi : ai);
            context.localBarrier();scores[lane]=score;ids[lane]=id;context.localBarrier();
        }
        if(lane==0) {
            double sum=0;for(int i=0;i<512;i++) sum+=(float)TornadoMath.exp((double)(logits.get(i)-scores[0]));
            double chosen=0;
            for(int i=0;i<selected.getSize();i++) {
                selected.set(i,ids[i]);float p=(float)TornadoMath.exp((double)(scores[i]-scores[0]));float w=(float)(p/sum);
                weights.set(i,w);chosen+=w;
            }
            if(normalize==1) for(int i=0;i<selected.getSize();i++) weights.set(i,(float)(weights.get(i)/chosen));
        }
    }
    public static void expertActivation(BFloat16Array upGate,FloatArray output,int hidden,int paddedBatch) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            int expert=i/(paddedBatch*hidden),row=i/hidden%paddedBatch,column=i%hidden;
            if(row==0) {
                int base=expert*paddedBatch*2*hidden;float up=value(upGate.get(base+column)),gate=value(upGate.get(base+hidden+column));
                output.set(i,bf((float)((double)gate/(1+TornadoMath.exp(-(double)gate)))*up));
            } else output.set(i,0);
        }
    }
    /** Each expert owns a separate 128x4 scaling tile even though decode pads only 32 rows. */
    public static void swizzleExperts(ByteArray input,ByteArray output,int columns,int paddedBatch) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            int groupBytes=128*columns,group=i/groupBytes,offset=i%groupBytes,tile=offset/512,local=offset%512;
            int row=(local%16)/4*32+local/16,column=tile*4+local%4;
            output.set(i,row<paddedBatch ? input.get(group*paddedBatch*columns+row*columns+column) : (byte)0);
        }
    }
    public static void finish(BFloat16Array down,FloatArray weights,FloatArray shared,FloatArray sharedRouter,FloatArray output,int paddedBatch) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            float sum=0;for(int expert=0;expert<weights.getSize();expert++) sum+=weights.get(expert)*value(down.get(expert*paddedBatch*output.getSize()+i));
            float gate=bf((float)(1/(1+TornadoMath.exp(-(double)sharedRouter.get(0)))));
            output.set(i,bf(bf(sum)+bf(shared.get(i)*gate)));
        }
    }
    public static void readWeights(FloatArray input,FloatArray output) {for(@Parallel int i=0;i<input.getSize();i++) output.set(i,input.get(i));}
    public static void seedEncoding(FloatArray e4,FloatArray e2,IntArray mode,FloatArray first,FloatArray second,FloatArray scratch) {
        for(@Parallel int i=0;i<scratch.getSize();i++) scratch.set(i,e4.get(0)+e2.get(0)+mode.get(1)+first.get(2)+second.get(2));
    }
}

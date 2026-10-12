package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** BF16 HC and PLE arithmetic; PLE convolution and checkpoints remain device resident. */
public final class FlashNextLayerKernels {
    private FlashNextLayerKernels() {}
    private static float bf(float value) {return FlashNextDeltaNetKernels.bf16(value);}
    private static float sigmoid(float value) {return (float)(1/(1+TornadoMath.exp(-(double)value)));}
    private static float silu(float value) {return bf((float)(value/(1+TornadoMath.exp(-(double)value))));}
    private static void normLane(FloatArray input,FloatArray weights,FloatArray output,int start,int dimension,float epsilon) {
        double sum=0;
        for(int j=0;j<dimension;j++) {float value=input.get(start+j);sum+=value*value;}
        float factor=(float)(1/TornadoMath.sqrt((double)((float)(sum/dimension)+epsilon)));
        for(int j=0;j<dimension;j++) output.set(start+j,bf(input.get(start+j)*factor*(1+weights.get(start+j))));
    }
    public static void groupedNorm(FloatArray input,FloatArray weights,FloatArray output,int dimension,float epsilon) {
        for(@Parallel int branch=0;branch<input.getSize()/dimension;branch++) normLane(input,weights,output,branch*dimension,dimension,epsilon);
    }
    public static void activateLow(FloatArray low,IntArray fused,int branches) {
        for(@Parallel int i=0;i<low.getSize();i++) {
            float value=fused.get(0)==1 ? low.get(i) : bf(low.get(i));
            low.set(i,silu(value/branches));
        }
    }
    public static void mix(FloatArray normalized,FloatArray gates,IntArray fused,FloatArray output,int branches) {
        for(@Parallel int j=0;j<output.getSize();j++) {
            float sum=0;
            for(int branch=0;branch<branches;branch++) {
                int offset=branch*output.getSize()+j;
                float gate=fused.get(0)==1 ? gates.get(offset) : bf(gates.get(offset));
                sum+=sigmoid(gate)*normalized.get(offset);
            }
            output.set(j,bf(sum/branches));
        }
    }
    public static void combine(FloatArray block,FloatArray residual,FloatArray injection,FloatArray output,int branches) {
        for(@Parallel int i=0;i<output.getSize();i++) {
            int branch=i/block.getSize();
            float gain=2*sigmoid(injection.get(branch)/branches);
            // TornadoVM's CUDA FMA phase fuses the multiply/add, matching staged HC combine.
            output.set(i,bf(block.get(i%block.getSize())*gain+residual.get(i)));
        }
    }
    public static void normalizePle(IntArray control,FloatArray key,FloatArray hidden,FloatArray keyWeights,
                                     FloatArray queryWeights,FloatArray keyOutput,FloatArray queryOutput,int dimension,float epsilon) {
        for(@Parallel int branch=0;branch<hidden.getSize()/dimension;branch++) {
            if(control.get(0)==0) {
                normLane(key,keyWeights,keyOutput,branch*dimension,dimension,epsilon);
                normLane(hidden,queryWeights,queryOutput,branch*dimension,dimension,epsilon);
            }
        }
    }
    public static void pleGate(IntArray control,FloatArray key,FloatArray query,FloatArray value,FloatArray gated,int dimension,float eagerClamp) {
        for(@Parallel int branch=0;branch<gated.getSize()/dimension;branch++) {
            if(control.get(0)==0) {
                float sum=0;
                for(int j=0;j<dimension;j++) sum+=bf(key.get(branch*dimension+j)*query.get(branch*dimension+j));
                float gate=bf(bf(sum)/(float)TornadoMath.sqrt((double)dimension));
                float clamp=control.get(3)==1 ? 1e-6f : eagerClamp;
                float magnitude=TornadoMath.max(TornadoMath.abs(gate),clamp);
                float root=bf((float)TornadoMath.sqrt((double)magnitude));
                float sign=gate>0 ? 1 : gate<0 ? -1 : 0;
                float activated=bf(sigmoid(bf(root*sign)));
                for(int j=0;j<dimension;j++) gated.set(branch*dimension+j,bf(activated*value.get(j)));
            }
        }
    }
    public static void normalizePleConvolution(IntArray control,FloatArray gated,FloatArray weights,FloatArray normalized,int dimension,float epsilon) {
        for(@Parallel int branch=0;branch<gated.getSize()/dimension;branch++)
            if(control.get(0)==0) normLane(gated,weights,normalized,branch*dimension,dimension,epsilon);
    }
    public static void pleConvolution(IntArray control,FloatArray normalized,FloatArray gated,FloatArray hidden,
                                       FloatArray weights,FloatArray state,FloatArray output,int kernel,int dilation) {
        for(@Parallel int channel=0;channel<output.getSize();channel++) {
            if(control.get(0)==0) {
                int history=(kernel-1)*dilation,start=channel*history;
                float sum=0;
                for(int tap=0;tap<kernel-1;tap++) sum+=state.get(start+tap*dilation)*weights.get(channel*kernel+tap);
                sum+=normalized.get(channel)*weights.get(channel*kernel+kernel-1);
                float added=bf(gated.get(channel)+silu(bf(sum)));
                output.set(channel,control.get(2)==1 ? bf(hidden.get(channel)+added) : added);
                for(int tap=0;tap<history-1;tap++) state.set(start+tap,state.get(start+tap+1));
                if(history>0) state.set(start+history-1,normalized.get(channel));
            }
        }
    }
}

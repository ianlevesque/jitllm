package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.math.TornadoMath;

/** BF16 activations, FP32 persistent recurrence, and device-only state checkpoints. */
public final class FlashNextDeltaNetKernels {
    private FlashNextDeltaNetKernels() {}

    public static float bf16(float value) {
        int bits=Float.floatToRawIntBits(value);
        if((bits&0x7f800000)==0x7f800000) return value;
        bits+=0x7fff+((bits>>>16)&1);
        return Float.intBitsToFloat(bits&0xffff0000);
    }
    /** action: 0=forward, 1=save, 2=restore, 3=reset. */
    public static void checkpoint(IntArray control,FloatArray state,FloatArray bank) {
        for(@Parallel int i=0;i<state.getSize();i++) {
            int action=control.get(0),offset=control.get(1)*state.getSize()+i;
            if(action==1) bank.set(offset,state.get(i));
            else if(action==2) state.set(i,bank.get(offset));
            else if(action==3) state.set(i,0);
        }
    }
    public static void convolve(IntArray control,FloatArray projected,FloatArray weights,FloatArray state,
                                FloatArray output,int kernel) {
        for(@Parallel int channel=0;channel<projected.getSize();channel++) {
            if(control.get(0)==0) {
                int history=kernel-1,start=channel*history;
                float sum=0;
                for(int tap=0;tap<history;tap++) sum+=state.get(start+tap)*weights.get(channel*kernel+tap);
                sum+=projected.get(channel)*weights.get(channel*kernel+kernel-1);
                output.set(channel,bf16((float)(sum/(1+TornadoMath.exp(-(double)sum)))));
                for(int tap=0;tap<history-1;tap++) state.set(start+tap,state.get(start+tap+1));
                if(history>0) state.set(start+history-1,projected.get(channel));
            }
        }
    }
    public static void normalizeQk(IntArray control,FloatArray convolved,FloatArray query,FloatArray key,int keyDim) {
        for(@Parallel int head=0;head<query.getSize()/keyDim;head++) {
            if(control.get(0)==0) {
                double qSum=0,kSum=0;
                for(int k=0;k<keyDim;k++) {
                    float q=convolved.get(head*keyDim+k),kv=convolved.get(query.getSize()+head*keyDim+k);
                    qSum+=q*q;kSum+=kv*kv;
                }
                float qScale=(float)(1/TornadoMath.sqrt((double)((float)qSum+1e-6f))/TornadoMath.sqrt((double)keyDim));
                float kScale=(float)(1/TornadoMath.sqrt((double)((float)kSum+1e-6f)));
                for(int k=0;k<keyDim;k++) {
                    query.set(head*keyDim+k,convolved.get(head*keyDim+k)*qScale);
                    key.set(head*keyDim+k,convolved.get(query.getSize()+head*keyDim+k)*kScale);
                }
            }
        }
    }
    public static void recur(IntArray control,FloatArray convolved,FloatArray query,FloatArray key,
                              FloatArray alpha,FloatArray beta,FloatArray aLog,FloatArray dtBias,
                              FloatArray state,FloatArray core,int keyDim,int valueDim,int stateOffset) {
        for(@Parallel int index=0;index<core.getSize();index++) {
            if(control.get(0)==0) {
                int head=index/valueDim,value=index%valueDim;
                int keyHead=head/((core.getSize()/valueDim)/(key.getSize()/keyDim));
                int base=stateOffset+head*keyDim*valueDim+value;
                float step=alpha.get(head)+dtBias.get(head);
                float softplus=step>20 ? step : (float)TornadoMath.log(1+TornadoMath.exp((double)step));
                float decay=(float)TornadoMath.exp((double)(-(float)TornadoMath.exp((double)aLog.get(head))*softplus));
                float write=(float)(1/(1+TornadoMath.exp(-(double)beta.get(head))));
                float correction=0;
                for(int k=0;k<keyDim;k++) {
                    int offset=base+k*valueDim;
                    float decayed=state.get(offset)*decay;
                    state.set(offset,decayed);
                    correction+=key.get(keyHead*keyDim+k)*decayed;
                }
                correction=(convolved.get(2*key.getSize()+index)-correction)*write;
                float result=0;
                for(int k=0;k<keyDim;k++) {
                    int offset=base+k*valueDim;
                    float updated=key.get(keyHead*keyDim+k)*correction+state.get(offset);
                    state.set(offset,updated);
                    result+=query.get(keyHead*keyDim+k)*updated;
                }
                core.set(index,bf16(result));
            }
        }
    }
    public static void gateNorm(IntArray control,FloatArray core,FloatArray gate,FloatArray weights,
                                 FloatArray output,int valueDim,float epsilon) {
        for(@Parallel int head=0;head<core.getSize()/valueDim;head++) {
            if(control.get(0)==0) {
                double squares=0;
                for(int v=0;v<valueDim;v++) {float value=core.get(head*valueDim+v);squares+=value*value;}
                float factor=(float)(1/TornadoMath.sqrt((double)((float)(squares/valueDim)+epsilon)));
                for(int v=0;v<valueDim;v++) {
                    int index=head*valueDim+v;
                    float sigmoid=(float)(1/(1+TornadoMath.exp(-(double)gate.get(index))));
                    output.set(index,bf16(core.get(index)*factor*weights.get(v)*sigmoid));
                }
            }
        }
    }
}

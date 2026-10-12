package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.Arrays;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;

/** Qwen4 Gated DeltaNet: BF16 projections/convolution and FP32 recurrent state. */
public final class Qwen4DeltaNet {
    private final Qwen4Configuration config;
    private final ModelOptMatrix qkv,z,a,b,out;
    private final float[] conv,norm,aLog,dtBias;

    public final class State {
        private final Qwen4DeltaNet owner=Qwen4DeltaNet.this;
        private final float[] convolution=new float[config.linearQkvDim()*(config.linearConvKernel()-1)];
        private final float[] temporal=new float[config.linearValueHeads()*config.linearKeyDim()*config.linearValueDim()];
        public record Snapshot(float[] convolution,float[] temporal) {}
        public Snapshot snapshot() { return new Snapshot(convolution.clone(),temporal.clone()); }
        public void restore(Snapshot snapshot) {
            if(snapshot.convolution().length!=convolution.length || snapshot.temporal().length!=temporal.length)
                throw new IllegalArgumentException("Wrong DeltaNet state shape");
            System.arraycopy(snapshot.convolution(),0,convolution,0,convolution.length);
            System.arraycopy(snapshot.temporal(),0,temporal,0,temporal.length);
        }
        public void reset() { Arrays.fill(convolution,0); Arrays.fill(temporal,0); }
    }

    public Qwen4DeltaNet(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix) throws IOException {
        this(config,index,reader,prefix,null);
    }
    public Qwen4DeltaNet(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,MatrixExecutor executor) throws IOException {
        this.config=config;
        int valueDim=config.linearValueHeads()*config.linearValueDim();
        qkv=Qwen4CpuLayers.matrix(index,reader,prefix+".in_proj_qkv",config.linearQkvDim(),config.dim(),executor);
        z=Qwen4CpuLayers.matrix(index,reader,prefix+".in_proj_z",valueDim,config.dim(),executor);
        a=Qwen4CpuLayers.matrix(index,reader,prefix+".in_proj_a",config.linearValueHeads(),config.dim(),executor);
        b=Qwen4CpuLayers.matrix(index,reader,prefix+".in_proj_b",config.linearValueHeads(),config.dim(),executor);
        out=Qwen4CpuLayers.matrix(index,reader,prefix+".out_proj",config.dim(),valueDim,executor);
        conv=Qwen4CpuLayers.vector(index,reader,prefix+".conv1d.weight",config.linearQkvDim()*config.linearConvKernel());
        norm=Qwen4CpuLayers.vector(index,reader,prefix+".norm.weight",config.linearValueDim());
        aLog=Qwen4CpuLayers.vector(index,reader,prefix+".A_log",config.linearValueHeads());
        dtBias=Qwen4CpuLayers.vector(index,reader,prefix+".dt_bias",config.linearValueHeads());
    }
    public State newState() { return new State(); }
    private static float bf(float value) { return FloatFormats.roundBf16(value); }

    public float[] forward(float[] input,State state) throws IOException {
        if(input.length!=config.dim() || state.owner!=this) throw new IllegalArgumentException("Invalid DeltaNet input/state");
        float[] projected=Qwen4CpuLayers.project(qkv,input),gates=Qwen4CpuLayers.project(z,input);
        float[] alpha=Qwen4CpuLayers.project(a,input),beta=Qwen4CpuLayers.project(b,input);
        int kernel=config.linearConvKernel(),history=kernel-1;
        float[] convolved=new float[projected.length];
        for(int channel=0; channel<projected.length; channel++) {
            float sum=0;
            int start=channel*history;
            for(int tap=0; tap<history; tap++) sum=Math.fma(state.convolution[start+tap],conv[channel*kernel+tap],sum);
            sum=Math.fma(projected[channel],conv[channel*kernel+kernel-1],sum);
            // The causal-convolution kernel applies SiLU before casting to its BF16 output.
            convolved[channel]=bf((float)(sum/(1+Math.exp(-sum))));
            if(history>0) {
                System.arraycopy(state.convolution,start+1,state.convolution,start,history-1);
                state.convolution[start+history-1]=projected[channel];
            }
        }
        int keyHeads=config.linearKeyHeads(),valueHeads=config.linearValueHeads();
        int kDim=config.linearKeyDim(),vDim=config.linearValueDim(),keyWidth=keyHeads*kDim;
        float[] query=new float[keyWidth],key=new float[keyWidth];
        for(int head=0; head<keyHeads; head++) {
            double querySum=0,keySum=0;
            for(int k=0; k<kDim; k++) {
                float q=convolved[head*kDim+k],kv=convolved[keyWidth+head*kDim+k];
                querySum+=q*q; keySum+=kv*kv;
            }
            float qScale=(float)(1/Math.sqrt((float)querySum+1e-6f)/Math.sqrt(kDim));
            float kScale=(float)(1/Math.sqrt((float)keySum+1e-6f));
            for(int k=0; k<kDim; k++) {
                query[head*kDim+k]=convolved[head*kDim+k]*qScale;
                key[head*kDim+k]=convolved[keyWidth+head*kDim+k]*kScale;
            }
        }
        float[] core=new float[valueHeads*vDim],correction=new float[vDim];
        for(int head=0; head<valueHeads; head++) {
            // GVA expands each key head to consecutive value heads, not modulo-interleaving.
            int keyHead=head/(valueHeads/keyHeads),base=head*kDim*vDim;
            float step=alpha[head]+dtBias[head];
            float softplus=step>20 ? step : (float)Math.log1p(Math.exp(step));
            float logDecay=-(float)Math.exp(aLog[head])*softplus;
            float decay=(float)Math.exp(logDecay);
            float write=(float)(1/(1+Math.exp(-beta[head])));
            Arrays.fill(correction,0);
            for(int k=0; k<kDim; k++) {
                for(int v=0; v<vDim; v++) {
                    int offset=base+k*vDim+v;
                    state.temporal[offset]*=decay;
                    correction[v]=Math.fma(key[keyHead*kDim+k],state.temporal[offset],correction[v]);
                }
            }
            for(int v=0; v<vDim; v++) correction[v]=(convolved[2*keyWidth+head*vDim+v]-correction[v])*write;
            for(int k=0; k<kDim; k++) {
                for(int v=0; v<vDim; v++) {
                    int offset=base+k*vDim+v;
                    state.temporal[offset]=Math.fma(key[keyHead*kDim+k],correction[v],state.temporal[offset]);
                    core[head*vDim+v]=Math.fma(query[keyHead*kDim+k],state.temporal[offset],core[head*vDim+v]);
                }
            }
        }
        float[] normalized=new float[core.length];
        for(int head=0; head<valueHeads; head++) {
            double squares=0;
            for(int v=0; v<vDim; v++) { core[head*vDim+v]=bf(core[head*vDim+v]); squares+=core[head*vDim+v]*core[head*vDim+v]; }
            float factor=(float)(1/Math.sqrt((float)(squares/vDim)+config.rmsNormEps()));
            for(int v=0; v<vDim; v++) {
                int offset=head*vDim+v;
                float gate=gates[offset];
                // Flash-Next output_gate_type is sigmoid, unlike the SiLU default
                // used by other Gated DeltaNet families.
                float gated=(float)(1/(1+Math.exp(-gate)));
                normalized[offset]=bf(core[offset]*factor*norm[v]*gated);
            }
        }
        return Qwen4CpuLayers.project(out,normalized);
    }
}

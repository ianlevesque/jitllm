package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;

/** Qwen4 top-k routing, NVFP4 W4A4 routed experts and the gated BF16 shared expert. */
public final class Qwen4Moe {
    public record Routing(int[] experts,float[] weights) {}
    public record Result(float[] output,Routing routing) {}
    private final Qwen4Configuration config;
    private final ModelOptMatrix router,sharedGate,sharedUp,sharedDown,sharedRouter;
    private final ModelOptMatrix[] gates,ups,downs;
    private final float inputScale,intermediateScale;

    public Qwen4Moe(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix) throws IOException {
        this(config,index,reader,prefix,null);
    }
    public Qwen4Moe(Qwen4Configuration config,SafeTensorIndex index,PagedTensorReader reader,String prefix,MatrixExecutor executor) throws IOException {
        this.config=config;
        router=Qwen4CpuLayers.matrix(index,reader,prefix+".gate",config.experts(),config.dim(),executor);
        sharedGate=Qwen4CpuLayers.matrix(index,reader,prefix+".shared_expert.gate_proj",config.sharedExpertDim(),config.dim(),executor);
        sharedUp=Qwen4CpuLayers.matrix(index,reader,prefix+".shared_expert.up_proj",config.sharedExpertDim(),config.dim(),executor);
        sharedDown=Qwen4CpuLayers.matrix(index,reader,prefix+".shared_expert.down_proj",config.dim(),config.sharedExpertDim(),executor);
        sharedRouter=Qwen4CpuLayers.matrix(index,reader,prefix+".shared_expert_gate",1,config.dim(),executor);
        gates=new ModelOptMatrix[config.experts()];ups=new ModelOptMatrix[config.experts()];downs=new ModelOptMatrix[config.experts()];
        float in=0,mid=0;
        for(int expert=0;expert<config.experts();expert++) {
            String name=prefix+".experts."+expert;
            gates[expert]=Qwen4CpuLayers.matrix(index,reader,name+".gate_proj",config.hiddenDim(),config.dim(),executor);
            ups[expert]=Qwen4CpuLayers.matrix(index,reader,name+".up_proj",config.hiddenDim(),config.dim(),executor);
            downs[expert]=Qwen4CpuLayers.matrix(index,reader,name+".down_proj",config.dim(),config.hiddenDim(),executor);
            if(!gates[expert].dtype().equals("U8") || !ups[expert].dtype().equals("U8") || !downs[expert].dtype().equals("U8"))
                throw new IOException("Target routed experts must use NVFP4: "+name);
            in=Math.max(in,Math.max(scale(index,reader,name+".gate_proj.input_scale"),scale(index,reader,name+".up_proj.input_scale")));
            mid=Math.max(mid,scale(index,reader,name+".down_proj.input_scale"));
        }
        inputScale=in;intermediateScale=mid;
    }
    private static float scale(SafeTensorIndex index,PagedTensorReader reader,String name) throws IOException {
        var tensor=index.require(name);
        if(!tensor.dtype().equals("F32") || tensor.elements()!=1) throw new IOException("Expected NVFP4 input scale scalar");
        float value=ByteBuffer.wrap(reader.read(tensor,0,4)).order(ByteOrder.LITTLE_ENDIAN).getFloat();
        if(!Float.isFinite(value) || value<=0) throw new IOException("Invalid NVFP4 input scale");
        return value;
    }
    public float inputScale() { return inputScale; }
    public float intermediateScale() { return intermediateScale; }
    private static float bf(float value) { return FloatFormats.roundBf16(value); }
    private static float[] swiglu(float[] gate,float[] up) {
        float[] activated=new float[gate.length];
        for(int i=0;i<gate.length;i++) activated[i]=bf((float)(gate[i]/(1+Math.exp(-gate[i])))*up[i]);
        return activated;
    }
    private static float[] projectPacked(ModelOptMatrix matrix,Nvfp4Activation.Packed input) throws IOException {
        float[] output=new float[matrix.rows()];matrix.multiplyNvfp4(input,output);
        for(int i=0;i<output.length;i++) output[i]=bf(output[i]);
        return output;
    }
    public Result forward(float[] input) throws IOException {
        if(input.length!=config.dim()) throw new IllegalArgumentException("Wrong MoE input width");
        float[] logits=Qwen4CpuLayers.project(router,input);
        Routing routing=route(logits,config.expertsPerToken(),config.normalizeTopK());
        var quantizedInput=Nvfp4Activation.quantize(input,inputScale);
        float[] accumulated=new float[config.dim()];
        for(int selected=0;selected<routing.experts().length;selected++) {
            int expert=routing.experts()[selected];
            float[] gate=projectPacked(gates[expert],quantizedInput),up=projectPacked(ups[expert],quantizedInput);
            float[] activated=swiglu(gate,up);
            var quantized=Nvfp4Activation.quantize(activated,intermediateScale);
            // CUTLASS UnfusedGemmOutputType is its BF16 backbone type;
            // routing finalization accumulates those rounded FC2 values in FP32.
            float[] output=projectPacked(downs[expert],quantized);
            for(int i=0;i<accumulated.length;i++) accumulated[i]=Math.fma(routing.weights()[selected],output[i],accumulated[i]);
        }
        float[] shared=Qwen4CpuLayers.project(sharedDown,swiglu(Qwen4CpuLayers.project(sharedGate,input),Qwen4CpuLayers.project(sharedUp,input)));
        float sharedWeight=bf((float)(1/(1+Math.exp(-Qwen4CpuLayers.project(sharedRouter,input)[0]))));
        for(int i=0;i<accumulated.length;i++) accumulated[i]=bf(bf(accumulated[i])+bf(shared[i]*sharedWeight));
        return new Result(accumulated,routing);
    }
    public static Routing route(float[] logits,int count,boolean normalize) {
        if(count<=0 || count>logits.length) throw new IllegalArgumentException("Invalid expert count");
        float maximum=Float.NEGATIVE_INFINITY;
        Integer[] order=new Integer[logits.length];
        for(int i=0;i<logits.length;i++) {
            if(!Float.isFinite(logits[i])) throw new IllegalArgumentException("Nonfinite router logit");
            maximum=Math.max(maximum,logits[i]);order[i]=i;
        }
        float[] probabilities=new float[logits.length];double sum=0;
        for(int i=0;i<logits.length;i++) {probabilities[i]=(float)Math.exp(logits[i]-maximum);sum+=probabilities[i];}
        Arrays.sort(order,Comparator.comparingDouble((Integer i)->-logits[i]).thenComparingInt(Integer::intValue));
        int[] ids=new int[count];float[] weights=new float[count];double selected=0;
        for(int i=0;i<count;i++) {ids[i]=order[i];weights[i]=(float)(probabilities[ids[i]]/sum);selected+=weights[i];}
        if(normalize) for(int i=0;i<count;i++) weights[i]=(float)(weights[i]/selected);
        return new Routing(ids,weights);
    }
}

package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.inference.op.QsaExecutor;

/** Stateful compressed QSA reference with causal sparse GQA and multimodal RoPE coordinates. */
public final class Qwen4QsaAttention {
    private final Qwen4Configuration config;
    private final ModelOptMatrix q,k,v,out,index;
    private final float[] qNorm,kNorm,indexQNorm,indexKNorm;
    private final boolean fp8Kv;
    private final boolean fusedPrecision;
    private final QsaExecutor executor;
    private final int draftTail;
    private final int[] ropeAxes;
    private final float[] frequencies;
    public final class State implements AutoCloseable {
        private final Qwen4QsaAttention owner=Qwen4QsaAttention.this;
        private final List<float[]> keys=new ArrayList<>(),values=new ArrayList<>(),compressed=new ArrayList<>();
        private final List<float[]> freshKeys=new ArrayList<>(),freshValues=new ArrayList<>();
        private final float[] pending=new float[config.indexerCompressRatio()*config.indexerHeadDim()];
        private final int[][] pendingPositions=new int[config.indexerCompressRatio()][3];
        private int[] selection=new int[0];
        private int prefillBlockSize=32;
        private int indexedThrough;
        private final QsaExecutor.Session device;
        private boolean closed;
        private State() {
            device=executor==null ? null : executor.newSession(new QsaExecutor.Shape(config.contextLength(),config.numberOfHeads(),config.numberOfKeyValueHeads(),config.headSize(),
                    config.indexerHeads(),config.indexerHeadDim(),config.indexerCompressRatio(),config.indexerBudget(),config.rotaryDimensions(),config.rmsNormEps(),draftTail),
                    qNorm,kNorm,indexQNorm,indexKNorm,ropeAxes,frequencies,fp8Kv,fusedPrecision);
        }
        private void requireOpen() {if(closed) throw new IllegalStateException("QSA state is closed");}
        public record Snapshot(State owner,int tokens,int blocks,float[] pending,int[][] positions,int[] selection,List<float[]> freshKeys,List<float[]> freshValues,int prefillBlockSize,int indexedThrough,QsaExecutor.Snapshot device) implements AutoCloseable {
            @Override public void close() {if(device!=null) device.close();}
        }
        public int tokens() { return device==null ? keys.size() : device.tokens(); }
        public int[] selectedTokens() {
            requireOpen();if(device!=null) try {return device.selectedTokens();} catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}
            return selection.clone();
        }
        public Snapshot snapshot() throws IOException {
            requireOpen();
            if(device!=null) return new Snapshot(this,tokens(),0,new float[0],new int[0][],new int[0],List.of(),List.of(),prefillBlockSize,0,device.snapshot());
            int[][] positions=Arrays.stream(pendingPositions).map(int[]::clone).toArray(int[][]::new);
            return new Snapshot(this,tokens(),compressed.size(),pending.clone(),positions,selection.clone(),List.copyOf(freshKeys),List.copyOf(freshValues),prefillBlockSize,indexedThrough,null);
        }
        public void restore(Snapshot snapshot) throws IOException {
            requireOpen();if(snapshot.owner()!=this) throw new IllegalArgumentException("Wrong QSA checkpoint owner");
            if(device!=null) {device.restore(snapshot.device());prefillBlockSize=snapshot.prefillBlockSize();return;}
            if(snapshot.tokens()<0 || snapshot.tokens()>tokens() || snapshot.blocks()>compressed.size()
                    || snapshot.pending().length!=pending.length || snapshot.positions().length!=pendingPositions.length)
                throw new IllegalArgumentException("Invalid QSA snapshot");
            keys.subList(snapshot.tokens(),keys.size()).clear(); values.subList(snapshot.tokens(),values.size()).clear();
            compressed.subList(snapshot.blocks(),compressed.size()).clear();
            System.arraycopy(snapshot.pending(),0,pending,0,pending.length);
            for(int i=0;i<pendingPositions.length;i++) System.arraycopy(snapshot.positions()[i],0,pendingPositions[i],0,3);
            selection=snapshot.selection().clone();
            freshKeys.clear();freshKeys.addAll(snapshot.freshKeys());freshValues.clear();freshValues.addAll(snapshot.freshValues());
            prefillBlockSize=snapshot.prefillBlockSize();
            indexedThrough=snapshot.indexedThrough();
        }
        /** SM120 uses the staged sparse kernel's L20 tuning table. */
        public void setPrefillQueryCount(int queries) {
            requireOpen();
            if(queries<=0) throw new IllegalArgumentException("Invalid prefill batch size");
            prefillBlockSize=queries<=32 ? 32 : queries<=128 ? 64 : queries<=512 ? 32 : 16;
        }
        public void finishFreshPrefill() {requireOpen();if(device!=null) device.finishFreshPrefill();else {freshKeys.clear();freshValues.clear();}}
        public void reset() {
            requireOpen();
            if(device!=null) {try {device.reset();} catch(IOException failure) {throw new java.io.UncheckedIOException(failure);}prefillBlockSize=32;return;}
            keys.clear(); values.clear(); compressed.clear(); Arrays.fill(pending,0);
            indexedThrough=0;
            finishFreshPrefill();
            for(int[] positions:pendingPositions) Arrays.fill(positions,0);
            selection=new int[0];
            prefillBlockSize=32;
        }
        @Override public void close() throws IOException {
            if(closed) return;closed=true;if(device!=null) device.close();
            keys.clear();values.clear();compressed.clear();freshKeys.clear();freshValues.clear();
        }
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix) throws IOException {
        this(config,tensors,reader,prefix,false);
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix,boolean fp8Kv) throws IOException {
        this(config,tensors,reader,prefix,fp8Kv,false);
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix,boolean fp8Kv,boolean fusedPrecision) throws IOException {
        this(config,tensors,reader,prefix,fp8Kv,fusedPrecision,null);
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix,boolean fp8Kv,boolean fusedPrecision,MatrixExecutor executor) throws IOException {
        this(config,tensors,reader,prefix,fp8Kv,fusedPrecision,executor,null);
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix,boolean fp8Kv,boolean fusedPrecision,MatrixExecutor matrices,QsaExecutor executor) throws IOException {
        this(config,tensors,reader,prefix,fp8Kv,fusedPrecision,matrices,executor,0);
    }
    public Qwen4QsaAttention(Qwen4Configuration config,SafeTensorIndex tensors,PagedTensorReader reader,String prefix,boolean fp8Kv,boolean fusedPrecision,MatrixExecutor matrices,QsaExecutor executor,int draftTail) throws IOException {
        if(draftTail<0 || draftTail>3) throw new IllegalArgumentException("Invalid MTP sparse tail capacity");
        this.config=config;
        this.draftTail=draftTail;
        this.fp8Kv=fp8Kv;
        this.fusedPrecision=fusedPrecision;
        this.executor=executor;
        ropeAxes=new int[config.rotaryDimensions()/2];frequencies=new float[ropeAxes.length];
        for(int i=0;i<ropeAxes.length;i++) {
            var sections=config.mropeSections();int axis=0;
            if(!sections.isEmpty()) {
                if(config.mropeInterleaved()) {if(i%3==1 && i<sections.get(1)*3) axis=1;else if(i%3==2 && i<sections.get(2)*3) axis=2;}
                else if(i>=sections.get(0)+sections.get(1)) axis=2;else if(i>=sections.get(0)) axis=1;
            }
            ropeAxes[i]=axis;frequencies[i]=(float)Math.pow(config.ropeTheta(),-2.0*i/config.rotaryDimensions());
        }
        q=Qwen4CpuLayers.matrix(tensors,reader,prefix+".q_proj",2*config.numberOfHeads()*config.headSize(),config.dim(),matrices);
        k=Qwen4CpuLayers.matrix(tensors,reader,prefix+".k_proj",config.kvDim(),config.dim(),matrices);
        v=Qwen4CpuLayers.matrix(tensors,reader,prefix+".v_proj",config.kvDim(),config.dim(),matrices);
        out=Qwen4CpuLayers.matrix(tensors,reader,prefix+".o_proj",config.dim(),config.numberOfHeads()*config.headSize(),matrices);
        index=Qwen4CpuLayers.matrix(tensors,reader,prefix+".indexer.index_qk_proj",(config.indexerHeads()+1)*config.indexerHeadDim(),config.dim(),matrices);
        qNorm=Qwen4CpuLayers.vector(tensors,reader,prefix+".q_norm.weight",config.headSize());
        kNorm=Qwen4CpuLayers.vector(tensors,reader,prefix+".k_norm.weight",config.headSize());
        indexQNorm=Qwen4CpuLayers.vector(tensors,reader,prefix+".indexer.q_layernorm.weight",config.indexerHeadDim());
        indexKNorm=Qwen4CpuLayers.vector(tensors,reader,prefix+".indexer.k_layernorm.weight",config.indexerHeadDim());
    }
    public State newState() { return new State(); }
    private static float bf(float value) { return FloatFormats.roundBf16(value); }
    private float[] normalize(float[] input,float[] weights) {
        float[] output=new float[input.length]; int width=weights.length;
        for(int start=0;start<input.length;start+=width) {
            double sum=0;for(int i=0;i<width;i++) sum+=input[start+i]*input[start+i];
            float factor=(float)(1/Math.sqrt((float)(sum/width)+config.rmsNormEps()));
            for(int i=0;i<width;i++) {
                float value=input[start+i]*factor*(1+weights[i]);
                output[start+i]=fusedPrecision ? value : bf(value);
            }
        }
        return output;
    }
    private void rope(float[] input,int width,int[] positions) {
        int half=config.rotaryDimensions()/2;
        for(int start=0;start<input.length;start+=width) for(int i=0;i<half;i++) {
            int axis=0;
            var sections=config.mropeSections();
            if(!sections.isEmpty()) {
                if(config.mropeInterleaved()) {
                    if(i%3==1 && i<sections.get(1)*3) axis=1;
                    else if(i%3==2 && i<sections.get(2)*3) axis=2;
                } else if(i>=sections.get(0)+sections.get(1)) axis=2;
                else if(i>=sections.get(0)) axis=1;
            }
            float frequency=(float)Math.pow(config.ropeTheta(),-2.0*i/config.rotaryDimensions());
            float angle=positions[axis]*frequency;
            float cosine=(float)Math.cos(angle),sine=(float)Math.sin(angle);
            if(!fusedPrecision) {cosine=bf(cosine);sine=bf(sine);}
            float left=input[start+i],right=input[start+half+i];
            input[start+i]=fusedPrecision ? bf(left*cosine-right*sine) : bf(bf(left*cosine)-bf(right*sine));
            input[start+half+i]=fusedPrecision ? bf(right*cosine+left*sine) : bf(bf(right*cosine)+bf(left*sine));
        }
        if(fusedPrecision) for(int start=0;start<input.length;start+=width)
            for(int i=config.rotaryDimensions();i<width;i++) input[start+i]=bf(input[start+i]);
    }
    public float[] forward(float[] hidden,State state) throws IOException {
        int position=state.tokens(); return forward(hidden,new int[]{position,position,position},state);
    }
    public float[] forward(float[] hidden,int[] ropePositions,State state) throws IOException {
        return forward(hidden,ropePositions,state,false);
    }
    public float[] forward(float[] hidden,State state,boolean freshPrefill) throws IOException {
        int position=state.tokens();return forward(hidden,new int[]{position,position,position},state,freshPrefill);
    }
    public float[] forward(float[] hidden,int[] ropePositions,State state,boolean freshPrefill) throws IOException {
        return forward(hidden,ropePositions,state,freshPrefill,freshPrefill);
    }
    public float[] forward(float[] hidden,State state,boolean prefill,boolean freshPrefill) throws IOException {
        int position=state.tokens();return forward(hidden,new int[]{position,position,position},state,prefill,freshPrefill);
    }
    public float[] forward(float[] hidden,int[] ropePositions,State state,boolean prefill,boolean freshPrefill) throws IOException {
        return forward(hidden,ropePositions,state,prefill,freshPrefill,null);
    }
    public float[] forwardFrozen(float[] hidden,State state,int[] selection) throws IOException {
        int position=state.tokens();return forward(hidden,new int[]{position,position,position},state,false,false,selection);
    }
    private float[] forward(float[] hidden,int[] ropePositions,State state,boolean prefill,boolean freshPrefill,int[] frozen) throws IOException {
        state.requireOpen();
        if(state.device==null && frozen==null && state.indexedThrough!=state.tokens()) throw new IllegalArgumentException("Restore indexed prefix before draft extend");
        if(freshPrefill && !prefill) throw new IllegalArgumentException("Fresh KV requires prefill");
        if(hidden.length!=config.dim() || state.owner!=this || ropePositions.length!=3 || state.tokens()>=config.contextLength())
            throw new IllegalArgumentException("Invalid QSA input/state/context");
        for(int position:ropePositions) if(position<0) throw new IllegalArgumentException("Negative RoPE coordinate");
        if(frozen!=null) {
            if(draftTail==0 || frozen.length==0 || frozen.length>config.indexerBudget()+config.indexerCompressRatio()-1+draftTail)
                throw new IllegalArgumentException("Invalid frozen sparse capacity");
            var seen=new HashSet<Integer>();for(int token:frozen) if(token<0 || token>state.tokens() || !seen.add(token)) throw new IllegalArgumentException("Invalid frozen sparse token");
        }
        if(state.device!=null && frozen!=null) return Qwen4CpuLayers.project(out,state.device.forwardFrozen(Qwen4CpuLayers.project(q,hidden),Qwen4CpuLayers.project(k,hidden),
                Qwen4CpuLayers.project(v,hidden),ropePositions,frozen));
        if(state.device!=null) return Qwen4CpuLayers.project(out,state.device.forward(Qwen4CpuLayers.project(q,hidden),Qwen4CpuLayers.project(k,hidden),
                Qwen4CpuLayers.project(v,hidden),Qwen4CpuLayers.project(index,hidden),ropePositions,prefill,freshPrefill,state.prefillBlockSize));
        int head=config.headSize(),heads=config.numberOfHeads();
        float[] projected=Qwen4CpuLayers.project(q,hidden),query=new float[heads*head],gate=new float[heads*head];
        for(int h=0;h<heads;h++) {
            System.arraycopy(projected,h*2*head,query,h*head,head);
            System.arraycopy(projected,h*2*head+head,gate,h*head,head);
        }
        query=normalize(query,qNorm); rope(query,head,ropePositions);
        float[] keys=normalize(Qwen4CpuLayers.project(k,hidden),kNorm); rope(keys,head,ropePositions);
        float[] values=Qwen4CpuLayers.project(v,hidden);
        if(freshPrefill) {
            if(state.freshKeys.size()!=state.tokens()) throw new IllegalArgumentException("Fresh prefill cannot follow cached-prefix attention");
            state.freshKeys.add(keys);state.freshValues.add(values);
            keys=keys.clone();values=values.clone();
        }
        if(fp8Kv) {
            for(int i=0;i<keys.length;i++) {
                int sign=(Float.floatToRawIntBits(keys[i])>>>24)&128;
                keys[i]=FloatFormats.e4m3(sign|Nvfp4Activation.e4m3(Math.abs(keys[i])));
            }
            for(int i=0;i<values.length;i++) {
                int sign=(Float.floatToRawIntBits(values[i])>>>24)&128;
                values[i]=FloatFormats.e4m3(sign|Nvfp4Activation.e4m3(Math.abs(values[i])));
            }
        }
        state.keys.add(keys); state.values.add(values);
        float[] indexQuery=null;
        int indexWidth=config.indexerHeadDim(),queryWidth=config.indexerHeads()*indexWidth;
        if(frozen==null) {
        float[] indexProjection=Qwen4CpuLayers.project(index,hidden);
        indexQuery=normalize(Arrays.copyOf(indexProjection,queryWidth),indexQNorm); rope(indexQuery,indexWidth,ropePositions);
        int slot=(state.tokens()-1)%config.indexerCompressRatio();
        System.arraycopy(indexProjection,queryWidth,state.pending,slot*indexWidth,indexWidth);
        state.pendingPositions[slot]=ropePositions.clone();
        if(state.tokens()%config.indexerCompressRatio()==0) {
            float[] pooled=new float[indexWidth];
            for(int i=0;i<indexWidth;i++) {
                float sum=0;for(int j=0;j<config.indexerCompressRatio();j++) sum+=state.pending[j*indexWidth+i];
                pooled[i]=fusedPrecision ? sum/config.indexerCompressRatio() : bf(sum/config.indexerCompressRatio());
            }
            pooled=normalize(pooled,indexKNorm);rope(pooled,indexWidth,state.pendingPositions[0]);
            state.compressed.add(pooled);
        }
        }
        state.selection=frozen==null ? select(indexQuery,state.compressed,state.tokens(),config.indexerHeads(),indexWidth,config.indexerBudget(),config.indexerCompressRatio()) : frozen.clone();
        if(frozen==null) state.indexedThrough=state.tokens();
        float[] attention=new float[query.length],scores=new float[state.selection.length];
        for(int h=0;h<heads;h++) {
            int keyHead=h/config.kvMul();float maximum=Float.NEGATIVE_INFINITY;
            for(int row=0;row<state.selection.length;row++) {
                float dot=0;float[] key=(freshPrefill ? state.freshKeys : state.keys).get(state.selection[row]);
                for(int i=0;i<head;i++) {
                    float qValue=query[h*head+i];
                    if(prefill && fusedPrecision) qValue=bf(qValue/(float)Math.sqrt(head)*1.4426950408f);
                    dot=Math.fma(qValue,key[keyHead*head+i],dot);
                }
                scores[row]=prefill && fusedPrecision ? dot : dot/(float)Math.sqrt(head);maximum=Math.max(maximum,scores[row]);
            }
            double denominator=0;
            if(prefill && fusedPrecision) {
                float onlineMax=Float.NEGATIVE_INFINITY,normalizer=0;
                for(int start=0;start<scores.length;start+=state.prefillBlockSize) {
                    int end=Math.min(start+state.prefillBlockSize,scores.length);float nextMax=onlineMax;
                    for(int row=start;row<end;row++) nextMax=Math.max(nextMax,scores[row]);
                    float alpha=(float)Math.pow(2,onlineMax-nextMax),sum=0;
                    for(int i=0;i<head;i++) attention[h*head+i]*=alpha;
                    for(int row=start;row<end;row++) {
                        float probability=(float)Math.pow(2,scores[row]-nextMax);sum+=probability;
                        float[] value=(freshPrefill ? state.freshValues : state.values).get(state.selection[row]);
                        for(int i=0;i<head;i++) attention[h*head+i]=Math.fma(bf(probability),value[keyHead*head+i],attention[h*head+i]);
                    }
                    normalizer=normalizer*alpha+sum;onlineMax=nextMax;
                }
                for(int i=0;i<head;i++) attention[h*head+i]/=normalizer;
                continue;
            }
            for(int row=0;row<scores.length;row++) {
                scores[row]=(float)Math.exp(scores[row]-maximum);
                denominator+=scores[row];
            }
            for(int row=0;row<scores.length;row++) {
                float probability=(float)(scores[row]/denominator);
                float[] value=(freshPrefill ? state.freshValues : state.values).get(state.selection[row]);
                for(int i=0;i<head;i++) attention[h*head+i]=Math.fma(probability,value[keyHead*head+i],attention[h*head+i]);
            }
        }
        for(int i=0;i<attention.length;i++) {
            float weight=(float)(1/(1+Math.exp(-gate[i])));
            // SGLang's CUDA fused_sigmoid_mul keeps the sigmoid in FP32.
            attention[i]=bf(bf(attention[i])*(fusedPrecision ? weight : bf(weight)));
        }
        return Qwen4CpuLayers.project(out,attention);
    }
    private record Block(int index,float score) {}
    public static int[] select(float[] query,List<float[]> compressed,int visible,int heads,int width,int budget,int ratio) {
        if(visible<1 || heads<1 || width<1 || budget<ratio || ratio<2 || budget%ratio!=0
                || query.length!=heads*width || compressed.size()!=visible/ratio) throw new IllegalArgumentException("Invalid QSA selection layout");
        List<Block> ranked=new ArrayList<>(compressed.size());
        for(int block=0;block<compressed.size();block++) {
            float[] key=compressed.get(block);if(key.length!=width) throw new IllegalArgumentException("Invalid compressed key width");
            float score=0;
            for(int h=0;h<heads;h++) {
                float dot=0;for(int i=0;i<width;i++) dot=Math.fma(query[h*width+i],key[i],dot);
                score+=Math.max(dot,0);
            }
            ranked.add(new Block(block,score/(float)Math.sqrt(width)));
        }
        ranked.sort(Comparator.comparingDouble((Block block)->-block.score()).thenComparingInt(Block::index));
        int blocks=Math.min(ranked.size(),budget/ratio),tail=visible%ratio;
        int[] tokens=new int[blocks*ratio+tail];int offset=0;
        for(int i=0;i<blocks;i++) for(int member=0;member<ratio;member++) tokens[offset++]=ranked.get(i).index()*ratio+member;
        for(int i=visible-tail;i<visible;i++) tokens[offset++]=i;
        return tokens;
    }
}

package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.*;
import org.beehive.jitllm.inference.CancellableTokenConsumer;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.junit.Test;
import static org.junit.Assert.*;

public class Qwen4SpeculativeSequenceTest {
    private static final int VOCAB=11;
    private static int advance(int hash,int token) {return (hash*3+token+1)%VOCAB;}
    private static Qwen4CpuEngine.Output output(int hash,int position,int bias) {
        float[] logits=new float[VOCAB];for(int i=0;i<VOCAB;i++) logits[i]=(float)Math.cos(i*.73+hash*.21);
        logits[(hash+bias)%VOCAB]=3;
        return new Qwen4CpuEngine.Output(new float[]{hash},new float[]{hash,position},logits);
    }
    private static Qwen4CpuEngine.Output masked(Qwen4CpuEngine.Output output,int valid) {
        Arrays.fill(output.logits(),valid,output.logits().length,Float.NEGATIVE_INFINITY);return output;
    }
    private abstract static class Endpoint implements Qwen4SpeculativeSequence.Endpoint {
        int position,hash,maxSnapshots,snapshotsTaken;boolean closed;final Set<Saved> saved=new HashSet<>();
        final class Saved implements Qwen4SpeculativeSequence.Checkpoint {
            final Endpoint owner=Endpoint.this;final int length=position,value=hash;boolean valid=true,disposed;
            public void close() {disposed=true;saved.remove(this);}
        }
        public int position() {return position;}
        public Qwen4SpeculativeSequence.Checkpoint snapshot() {var s=new Saved();saved.add(s);snapshotsTaken++;maxSnapshots=Math.max(maxSnapshots,saved.size());return s;}
        public void restore(Qwen4SpeculativeSequence.Checkpoint value) {
            var s=(Saved)value;if(s.owner!=this || !s.valid || s.disposed || s.length>position) throw new IllegalArgumentException("Invalid checkpoint");
            for(var future:saved) if(future.length>s.length) future.valid=false;position=s.length;hash=s.value;
        }
        public void reset() {position=hash=0;for(var s:List.copyOf(saved)) s.close();}
        public void beginPrefill(int n) {if(n<=0) throw new IllegalArgumentException();}
        public void finishPrefill() {}
        public void close() {closed=true;assertTrue(saved.isEmpty());}
    }
    private static final class Target extends Endpoint implements Qwen4SpeculativeSequence.Target {
        int failAt=-1,valid=VOCAB;
        public Qwen4CpuEngine.Output forward(int token,boolean logits) throws IOException {
            if(position==failAt) {failAt=-1;throw new IOException("injected target");}
            hash=advance(hash,token);position++;return masked(output(hash,position,0),valid);
        }
    }
    private static final class Draft extends Endpoint implements Qwen4SpeculativeSequence.Draft {
        final int bias;int valid=VOCAB;boolean failTeacher;
        Draft(int bias) {this.bias=bias;}
        public Qwen4CpuEngine.Output teacher(int token,float[] hidden,boolean logits) throws IOException {
            if(failTeacher) {failTeacher=false;throw new IOException("injected teacher");}
            assertEquals(position+1,(int)hidden[1]);hash=advance((int)hidden[0],token);position++;return masked(output(hash,position+1,bias),valid);
        }
        public Qwen4CpuEngine.Output propose(int token,float[] hidden) {hash=advance((int)hidden[0],token);position++;return masked(output(hash,position+1,bias),valid);}
    }
    private static List<Integer> baseline(List<Integer> prompt,int count,Sampler sampler) throws IOException {
        return baseline(prompt,count,sampler,VOCAB);
    }
    private static List<Integer> baseline(List<Integer> prompt,int count,Sampler sampler,int valid) throws IOException {
        var target=new Target();target.valid=valid;Qwen4CpuEngine.Output out=null;for(int token:prompt) out=target.forward(token,true);
        var result=new ArrayList<Integer>();for(int i=0;i<count;i++) {int token=sampler.sampleToken(Logits.of(new ArrayFloatTensor(out.logits().clone())));result.add(token);out=target.forward(token,true);}return result;
    }
    @Test public void acceptsAndCorrectsAgainstTheSameTarget() throws Exception {
        for(int bias:new int[]{0,1,4}) {
            var target=new Target();var draft=new Draft(bias);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
                var tokens=sequence.generate(0,List.of(2,3,4),Set.of(),18,Sampler.TENSOR_ARGMAX,null);
                assertEquals(baseline(List.of(2,3,4),15,Sampler.TENSOR_ARGMAX),tokens);assertEquals(18,sequence.position());assertEquals(17,target.position());
                assertTrue(target.maxSnapshots<=6);assertEquals(0,target.saved.size());assertEquals(0,draft.saved.size());
                if(bias==0) assertTrue(sequence.statistics().accepted()>0);else assertTrue(sequence.statistics().corrections()>0);
            }
            assertTrue(target.closed && draft.closed);
        }
    }
    @Test public void stochasticTargetSamplingPreservesTheSeededBaseline() throws Exception {
        for(int seed=0;seed<32;seed++) {
            var target=new Target();var draft=new Draft(seed%2);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
                assertEquals(baseline(List.of(2,3),20,Sampler.selectSampler(VOCAB,.9f,.8f,seed)),
                        sequence.generate(0,List.of(2,3),Set.of(),22,Sampler.selectSampler(VOCAB,.9f,.8f,seed),null));
            }
        }
    }
    @Test public void paddedHeadSamplingUsesTheHeadWidthAndMasksUnavailableTokens() throws Exception {
        for(int seed=0;seed<4;seed++) {
            int valid=VOCAB-2;var target=new Target();target.valid=valid;var draft=new Draft(1);draft.valid=valid;
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,valid,null)) {
                var tokens=sequence.generate(0,List.of(2,3),Set.of(),22,Sampler.selectSampler(VOCAB,.8f,.9f,seed),null);
                assertEquals(baseline(List.of(2,3),20,Sampler.selectSampler(VOCAB,.8f,.9f,seed),valid),tokens);
                assertTrue(tokens.stream().allMatch(token->token<valid));
            }
        }
    }
    @Test public void budgetContextAndCancellationRestoreOnlyTheCommittedPrefix() throws Exception {
        for(int budget=1;budget<9;budget++) {
            var target=new Target();var draft=new Draft(0);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,2+budget,VOCAB,null)) {
                assertEquals(baseline(List.of(1,2),budget,Sampler.TENSOR_ARGMAX),sequence.generate(0,List.of(1,2),Set.of(),2+budget,Sampler.TENSOR_ARGMAX,null));
                assertEquals(1+budget,target.position());assertEquals(2+budget,sequence.position());
            }
        }
        var target=new Target();var draft=new Draft(0);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,32,VOCAB,null)) {
            var consumer=new CancellableTokenConsumer() {int n;public void accept(int token) {n++;}public boolean cancellationRequested() {return n==3;}};
            assertEquals(3,sequence.generate(0,List.of(1,2),Set.of(),20,Sampler.TENSOR_ARGMAX,consumer).size());
            assertEquals(5,sequence.position());assertEquals(4,target.position());
        }
    }
    @Test public void multiTurnPendingTokensAndStopsDoNotPolluteThePrefix() throws Exception {
        for(boolean includeStop:new boolean[]{true,false}) {
            var target=new Target();var draft=new Draft(0);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
                var first=sequence.generate(0,List.of(2,3),Set.of(advance(advance(0,2),3)),20,Sampler.TENSOR_ARGMAX,null);
                assertEquals(1,first.size());assertEquals(2,sequence.position());assertEquals(2,target.position());
                var nextPrompt=includeStop ? List.of(first.getFirst(),4,5) : List.of(4,5);
                var full=new ArrayList<Integer>(List.of(2,3));full.addAll(nextPrompt);
                assertEquals(baseline(full,7,Sampler.TENSOR_ARGMAX),sequence.generate(2,nextPrompt,Set.of(),full.size()+7,Sampler.TENSOR_ARGMAX,null));
            }
        }
        var target=new Target();var draft=new Draft(0);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
            var first=sequence.generate(0,List.of(2,3),Set.of(),6,Sampler.TENSOR_ARGMAX,null);
            var full=new ArrayList<Integer>(List.of(2,3));full.addAll(first);full.add(7);
            assertEquals(baseline(full,5,Sampler.TENSOR_ARGMAX),sequence.generate(6,List.of(7),Set.of(),12,Sampler.TENSOR_ARGMAX,null));
            sequence.reset();assertEquals(0,sequence.position());assertEquals(first,sequence.generate(0,List.of(2,3),Set.of(),6,Sampler.TENSOR_ARGMAX,null));
        }
    }
    @Test public void verificationSkipsTheIdentityFinalPrefixCheckpoint() throws Exception {
        var target=new Target();var draft=new Draft(0);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
            var tokens=sequence.generate(0,List.of(2,3,4),Set.of(),18,Sampler.TENSOR_ARGMAX,null);
            assertEquals(baseline(List.of(2,3,4),15,Sampler.TENSOR_ARGMAX),tokens);
            var stats=sequence.statistics();
            assertTrue(stats.rounds()>0);
            // One snapshot from ingest, then per round one base snapshot plus one prefix
            // snapshot per proposed candidate; none after the final verified forward.
            assertEquals(1+stats.rounds()+stats.proposed(),target.snapshotsTaken);
            assertTrue(stats.checkpointsSkipped()>=stats.rounds());
            assertEquals(0,target.saved.size());
        }
    }
    @Test public void targetAndDeferredTeacherFailuresRetryWithoutAnotherSamplerDraw() throws Exception {
        var target=new Target();var draft=new Draft(0);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
            target.failAt=1;assertThrows(IOException.class,()->sequence.generate(0,List.of(2,3),Set.of(),10,Sampler.TENSOR_ARGMAX,null));
            assertEquals(0,sequence.position());assertEquals(0,target.position());assertEquals(0,draft.position());
            sequence.generate(0,List.of(2,3),Set.of(),3,Sampler.TENSOR_ARGMAX,null);
            draft.failTeacher=true;int[] draws={0};Sampler sampler=logits->{draws[0]++;return Sampler.TENSOR_ARGMAX.sampleToken(logits);};
            assertThrows(IOException.class,()->sequence.generate(3,List.of(4),Set.of(),10,sampler,null));assertEquals(0,draws[0]);assertEquals(3,sequence.position());
            var full=new ArrayList<Integer>(List.of(2,3));full.add(baseline(List.of(2,3),1,Sampler.TENSOR_ARGMAX).getFirst());full.add(4);
            assertEquals(baseline(full,6,Sampler.TENSOR_ARGMAX),sequence.generate(3,List.of(4),Set.of(),10,sampler,null));
        }
    }
}

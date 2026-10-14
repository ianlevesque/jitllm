package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.*;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Chunked speculative ingest against synthetic endpoints: the same hash-chained
 * target/draft pair as {@code Qwen4SpeculativeSequenceTest}, with a target that can
 * forward whole prefill chunks. The draft asserts every teacher feed carries the
 * hyper-hidden of the target output that precedes its token, so a mis-paired chunk
 * feed fails here even when the generated tokens would still match.
 */
public class Qwen4SpeculativeBatchedIngestTest {
    private static final int VOCAB=11;
    private static int advance(int hash,int token) {return (hash*3+token+1)%VOCAB;}
    private static Qwen4CpuEngine.Output output(int hash,int position,int bias,boolean logits) {
        float[] values=null;
        if(logits) {values=new float[VOCAB];for(int i=0;i<VOCAB;i++) values[i]=(float)Math.cos(i*.73+hash*.21);values[(hash+bias)%VOCAB]=3;}
        return new Qwen4CpuEngine.Output(new float[]{hash},new float[]{hash,position},values);
    }
    private abstract static class Endpoint implements Qwen4SpeculativeSequence.Endpoint {
        int position,hash,begins,finishes;final List<Integer> beginSizes=new ArrayList<>();boolean closed;final Set<Saved> saved=new HashSet<>();
        final List<String> events;Endpoint(List<String> events) {this.events=events;}
        final class Saved implements Qwen4SpeculativeSequence.Checkpoint {
            final Endpoint owner=Endpoint.this;final int length=position,value=hash;boolean valid=true,disposed;
            public void close() {disposed=true;saved.remove(this);}
        }
        public int position() {return position;}
        public Qwen4SpeculativeSequence.Checkpoint snapshot() {var s=new Saved();saved.add(s);return s;}
        public void restore(Qwen4SpeculativeSequence.Checkpoint value) {
            var s=(Saved)value;if(s.owner!=this || !s.valid || s.disposed || s.length>position) throw new IllegalArgumentException("Invalid checkpoint");
            for(var future:saved) if(future.length>s.length) future.valid=false;position=s.length;hash=s.value;
        }
        public void reset() {position=hash=0;for(var s:List.copyOf(saved)) s.close();}
        public void beginPrefill(int n) {if(n<=0) throw new IllegalArgumentException();begins++;beginSizes.add(n);}
        public void finishPrefill() {finishes++;events.add("FIN");}
        public void close() {closed=true;assertTrue(saved.isEmpty());}
    }
    /** Base target: keeps the interface's sequential default chunk implementation. */
    private static class Target extends Endpoint implements Qwen4SpeculativeSequence.Target {
        final boolean batched;final int rows;int failAt=-1;
        Target(List<String> events,boolean batched,int rows) {super(events);this.batched=batched;this.rows=rows;}
        Qwen4CpuEngine.Output step(int token,boolean logits) throws IOException {
            if(position==failAt) {failAt=-1;throw new IOException("injected target");}
            hash=advance(hash,token);position++;return output(hash,position,0,logits);
        }
        public Qwen4CpuEngine.Output forward(int token,boolean logits) throws IOException {events.add("F"+token);return step(token,true);}
        public boolean batchedPrefill() {return batched;}
        public int batchedPrefillRows() {return rows;}
    }
    /** Target with a real chunk override, honoring the engine's whole-chunk rollback. */
    private static final class ChunkTarget extends Target {
        final List<Integer> chunks=new ArrayList<>();
        ChunkTarget(List<String> events,boolean batched,int rows) {super(events,batched,rows);}
        @Override public Qwen4CpuEngine.Output[] forwardPrefillChunk(int[] tokens,boolean withLogits) throws IOException {
            chunks.add(tokens.length);events.add("C"+tokens.length);
            int savedHash=hash,savedPosition=position;
            try {
                var outputs=new Qwen4CpuEngine.Output[tokens.length];
                for(int i=0;i<tokens.length;i++) outputs[i]=step(tokens[i],withLogits && i==tokens.length-1);
                return outputs;
            } catch(IOException failure) {hash=savedHash;position=savedPosition;throw failure;}
        }
    }
    private static final class Draft extends Endpoint implements Qwen4SpeculativeSequence.Draft {
        final int bias;boolean failTeacher;
        Draft(List<String> events,int bias) {super(events);this.bias=bias;}
        public Qwen4CpuEngine.Output teacher(int token,float[] hidden,boolean logits) throws IOException {
            if(failTeacher) {failTeacher=false;throw new IOException("injected teacher");}
            events.add("T"+token);assertEquals(position+1,(int)hidden[1]);
            hash=advance((int)hidden[0],token);position++;return output(hash,position+1,bias,true);
        }
        public Qwen4CpuEngine.Output propose(int token,float[] hidden) {hash=advance((int)hidden[0],token);position++;return output(hash,position+1,bias,true);}
    }
    private static List<Integer> baseline(List<Integer> prompt,int count,Sampler sampler) throws IOException {
        var target=new Target(new ArrayList<>(),false,32);Qwen4CpuEngine.Output out=null;for(int token:prompt) out=target.forward(token,true);
        var result=new ArrayList<Integer>();for(int i=0;i<count;i++) {int token=sampler.sampleToken(Logits.of(new ArrayFloatTensor(out.logits().clone())));result.add(token);out=target.forward(token,true);}return result;
    }
    private static List<Integer> prompt(int count) {var tokens=new ArrayList<Integer>();for(int i=0;i<count;i++) tokens.add((i*3+1)%VOCAB);return tokens;}

    @Test public void chunkedIngestMatchesSequentialTokensStatisticsAndPrefillCounts() throws Exception {
        for(int rows:new int[]{1,3,32}) for(int bias:new int[]{0,1}) {
            var sequentialEvents=new ArrayList<String>();
            var seqTarget=new ChunkTarget(sequentialEvents,false,32);var seqDraft=new Draft(sequentialEvents,bias);
            List<Integer> expected;Qwen4SpeculativeSequence.Statistics expectedStats;
            try(var sequence=new Qwen4SpeculativeSequence(seqTarget,seqDraft,64,VOCAB,null)) {
                expected=sequence.generate(0,prompt(11),Set.of(),23,Sampler.TENSOR_ARGMAX,null);expectedStats=sequence.statistics();
            }
            assertEquals(baseline(prompt(11),12,Sampler.TENSOR_ARGMAX),expected);
            var events=new ArrayList<String>();
            var target=new ChunkTarget(events,true,rows);var draft=new Draft(events,bias);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
                assertEquals(expected,sequence.generate(0,prompt(11),Set.of(),23,Sampler.TENSOR_ARGMAX,null));
                assertEquals(expectedStats,sequence.statistics());
                assertEquals(23,sequence.position());assertEquals(22,target.position());
            }
            assertEquals(rows==1 ? List.of(1,1,1,1,1,1,1,1,1,1,1) : rows==3 ? List.of(3,3,3,2) : List.of(11),target.chunks);
            // One begin/finish pair per endpoint for the whole ingest, however many chunks.
            assertEquals(1,target.begins);assertEquals(1,target.finishes);assertEquals(List.of(11),target.beginSizes);
            assertEquals(1,draft.begins);assertEquals(1,draft.finishes);assertEquals(List.of(11),draft.beginSizes);
            assertEquals(seqTarget.finishes,target.finishes);assertEquals(seqDraft.finishes,draft.finishes);
            assertTrue(target.closed && draft.closed);
        }
    }

    @Test public void offModeNeverChunksAndInterleavesTeachers() throws Exception {
        var sequential=new ArrayList<String>();
        var offTarget=new ChunkTarget(sequential,false,32);var offDraft=new Draft(sequential,0);
        try(var sequence=new Qwen4SpeculativeSequence(offTarget,offDraft,64,VOCAB,null)) {
            sequence.generate(0,List.of(2,3,4),Set.of(),6,Sampler.TENSOR_ARGMAX,null);
        }
        assertTrue(offTarget.chunks.isEmpty());
        // Teacher for a token runs before that token's target forward, as today.
        assertEquals(List.of("F2","T3","F3","T4","F4"),sequential.subList(0,sequential.indexOf("FIN")));
        var batched=new ArrayList<String>();
        var onTarget=new ChunkTarget(batched,true,32);var onDraft=new Draft(batched,0);
        try(var sequence=new Qwen4SpeculativeSequence(onTarget,onDraft,64,VOCAB,null)) {
            sequence.generate(0,List.of(2,3,4),Set.of(),6,Sampler.TENSOR_ARGMAX,null);
        }
        // One chunk forward, then the same teachers in the same order.
        assertEquals(List.of("C3","T3","T4"),batched.subList(0,batched.indexOf("FIN")));
    }

    @Test public void defaultChunkImplementationIngestsSequentially() throws Exception {
        var events=new ArrayList<String>();
        var target=new Target(events,true,3);var draft=new Draft(events,1);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
            assertEquals(baseline(prompt(8),9,Sampler.TENSOR_ARGMAX),
                    sequence.generate(0,prompt(8),Set.of(),17,Sampler.TENSOR_ARGMAX,null));
        }
        assertEquals(1,target.finishes);assertEquals(1,draft.finishes);
    }

    @Test public void failureMidChunkRollsBackWholeIngestAndRetries() throws Exception {
        var events=new ArrayList<String>();
        var target=new ChunkTarget(events,true,3);var draft=new Draft(events,0);
        try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
            // Position 4 is inside the second chunk of three.
            target.failAt=4;
            assertThrows(IOException.class,()->sequence.generate(0,prompt(8),Set.of(),17,Sampler.TENSOR_ARGMAX,null));
            assertEquals(0,sequence.position());assertEquals(0,target.position());assertEquals(0,draft.position());
            assertEquals(1,target.begins);assertEquals(0,target.finishes);assertEquals(0,draft.finishes);
            assertEquals(0,target.saved.size());assertEquals(0,draft.saved.size());
            draft.failTeacher=true;
            assertThrows(IOException.class,()->sequence.generate(0,prompt(8),Set.of(),17,Sampler.TENSOR_ARGMAX,null));
            assertEquals(0,sequence.position());assertEquals(0,target.position());assertEquals(0,draft.position());
            assertEquals(0,target.finishes);
            assertEquals(baseline(prompt(8),9,Sampler.TENSOR_ARGMAX),
                    sequence.generate(0,prompt(8),Set.of(),17,Sampler.TENSOR_ARGMAX,null));
            assertEquals(3,target.begins);assertEquals(1,target.finishes);assertEquals(1,draft.finishes);
            assertTrue(sequence.statistics().rollbacks()>=2);
        }
    }

    @Test public void multiTurnLaggingIngestAndStochasticSamplingMatchBaseline() throws Exception {
        for(int seed=0;seed<8;seed++) {
            var events=new ArrayList<String>();
            var target=new ChunkTarget(events,true,3);var draft=new Draft(events,seed%2);
            try(var sequence=new Qwen4SpeculativeSequence(target,draft,64,VOCAB,null)) {
                assertEquals(baseline(List.of(2,3),4,Sampler.selectSampler(VOCAB,.9f,.8f,seed)),
                        sequence.generate(0,List.of(2,3),Set.of(),6,Sampler.selectSampler(VOCAB,.9f,.8f,seed),null));
                var full=new ArrayList<Integer>(List.of(2,3));
                full.addAll(baseline(List.of(2,3),4,Sampler.selectSampler(VOCAB,.9f,.8f,seed)));
                var turn=prompt(5);full.addAll(turn);
                assertEquals(baseline(full,5,Sampler.selectSampler(VOCAB,.9f,.8f,seed+100)),
                        sequence.generate(6,turn,Set.of(),16,Sampler.selectSampler(VOCAB,.9f,.8f,seed+100),null));
            }
        }
    }
}

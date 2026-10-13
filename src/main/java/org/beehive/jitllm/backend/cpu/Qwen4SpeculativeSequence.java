package org.beehive.jitllm.backend.cpu;

import java.io.IOException;
import java.util.*;
import java.util.function.IntConsumer;
import org.beehive.jitllm.inference.CancellableTokenConsumer;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Qwen4SequenceState;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;

/** Top-1 MTP tree verification with target sampling and request-owned prefix rollback.
 * Draft proposals consume no sampler draws; each emitted token is sampled once from
 * its target distribution. This is the staged top-1 tree-sampling policy, rather
 * than the optional full-distribution p/q rejection-sampling policy.
 */
public final class Qwen4SpeculativeSequence implements Qwen4SequenceState {
    public interface Checkpoint extends AutoCloseable {@Override void close();}
    public interface Endpoint extends AutoCloseable {
        int position();
        Checkpoint snapshot() throws IOException;
        void restore(Checkpoint saved) throws IOException;
        void reset();
        void beginPrefill(int tokens);
        void finishPrefill();
        @Override void close() throws IOException;
    }
    public interface Target extends Endpoint {Qwen4CpuEngine.Output forward(int token,boolean logits) throws IOException;}
    public interface Draft extends Endpoint {
        Qwen4CpuEngine.Output teacher(int successor,float[] targetHidden,boolean logits) throws IOException;
        Qwen4CpuEngine.Output propose(int successor,float[] draftHidden) throws IOException;
    }
    public record Statistics(long proposed,long accepted,long corrections,long rounds,long targetForwards,long teacherForwards,long draftForwards,long rollbacks) {}
    private record Feed(int successor,float[] hidden) {}
    private final Target target;
    private final Draft draft;
    private final int context,vocabulary;
    private final Runnable onClose;
    private final List<Feed> refresh=new ArrayList<>();
    private Qwen4CpuEngine.Output targetOutput,draftOutput;
    private int logical,pending=-1;
    private boolean pendingStop,finishTeacherPrefill,closed;
    private long proposed,accepted,corrections,rounds,targetForwards,teacherForwards,draftForwards,rollbacks;

    public Qwen4SpeculativeSequence(Target target,Draft draft,int context,int vocabulary,Runnable onClose) {
        if(context<=0 || vocabulary<=0 || target.position()!=0 || draft.position()!=0) throw new IllegalArgumentException("Invalid speculative endpoints");
        this.target=Objects.requireNonNull(target);this.draft=Objects.requireNonNull(draft);this.context=context;this.vocabulary=vocabulary;this.onClose=onClose;
    }
    private void check() {if(closed) throw new IllegalStateException("Speculative sequence is closed");}
    @Override public int position() {return logical;}
    public Statistics statistics() {return new Statistics(proposed,accepted,corrections,rounds,targetForwards,teacherForwards,draftForwards,rollbacks);}
    private Qwen4CpuEngine.Output targetForward(int token,boolean logits) throws IOException {var out=target.forward(token,logits);targetForwards++;return out;}
    private Qwen4CpuEngine.Output teacher(int token,float[] hidden,boolean logits) throws IOException {var out=draft.teacher(token,hidden,logits);teacherForwards++;return out;}
    private int sample(Qwen4CpuEngine.Output output,Sampler sampler) {
        int token=sampler.sampleToken(Logits.of(new ArrayFloatTensor(output.logits().clone())));
        if(token<0 || token>=vocabulary) throw new IllegalArgumentException("Sampler returned an invalid token");return token;
    }
    private boolean emit(int token,Set<Integer> stops,List<Integer> output,IntConsumer callback) {
        pending=token;pendingStop=stops.contains(token);if(!pendingStop) logical++;
        output.add(token);if(callback!=null) callback.accept(token);
        return pendingStop || callback instanceof CancellableTokenConsumer consumer && consumer.cancellationRequested();
    }
    /** Teacher refresh is performed before any new sampler draw, making retry deterministic. */
    private void refreshTeacher() throws IOException {
        if(refresh.isEmpty()) return;
        var previous=draftOutput;
        try(var before=draft.snapshot()) {
            try {
                for(int i=0;i<refresh.size();i++) {var feed=refresh.get(i);draftOutput=teacher(feed.successor,feed.hidden,i==refresh.size()-1);}
                if(finishTeacherPrefill) {draft.finishPrefill();finishTeacherPrefill=false;}
                refresh.clear();
            } catch(IOException|RuntimeException|Error failure) {draftOutput=previous;try {draft.restore(before);rollbacks++;} catch(Exception rollback) {failure.addSuppressed(rollback);}throw failure;}
        }
    }
    public List<Integer> generate(int start,List<Integer> prompt,Set<Integer> stops,int bound,Sampler sampler,IntConsumer callback) throws IOException {
        check();Objects.requireNonNull(sampler);Objects.requireNonNull(stops);
        if(start!=logical || bound<logical || bound>context || (long)logical+prompt.size()>bound) throw new IllegalArgumentException("Invalid speculative position/budget");
        for(int token:prompt) if(token<0 || token>=vocabulary) throw new IllegalArgumentException("Invalid prompt token");
        // A terminal stop is not part of the API's cached prefix. Only keep its
        // shifted teacher entry when the following prompt actually starts with it.
        if(pendingStop && (prompt.isEmpty() || prompt.getFirst()!=pending)) {
            if(refresh.isEmpty() || refresh.getLast().successor!=pending) throw new IllegalStateException("Missing deferred stop entry");
            refresh.removeLast();pending=-1;pendingStop=false;
        }
        refreshTeacher();
        if(pending>=0 && !pendingStop) {
            var previous=targetOutput;
            try(var before=target.snapshot()) {
                try {targetOutput=targetForward(pending,true);pending=-1;}
                catch(IOException|RuntimeException|Error failure) {targetOutput=previous;try {target.restore(before);rollbacks++;} catch(Exception rollback) {failure.addSuppressed(rollback);}throw failure;}
            }
        }
        if(!prompt.isEmpty()) ingest(prompt);
        if(target.position()!=logical) throw new IllegalStateException("Target prefix does not match the API position");
        var output=new ArrayList<Integer>();if(logical==bound) return output;
        if(targetOutput==null) throw new IllegalArgumentException("Initial generation needs a prompt");
        int seed=sample(targetOutput,sampler);
        refresh.add(new Feed(seed,targetOutput.hyperHidden()));
        if(emit(seed,stops,output,callback) || logical==bound) return output;
        while(logical<bound) {
            refreshTeacher();
            if(verifyRound(bound,stops,sampler,callback,output)) break;
        }
        return output;
    }
    private void ingest(List<Integer> prompt) throws IOException {
        int teachers=prompt.size()+(draft.position()==target.position()-1 ? 1 : 0);
        int previousLogical=logical,previousPending=pending;boolean previousStop=pendingStop;var previousTarget=targetOutput;var previousDraft=draftOutput;
        try(var targetBefore=target.snapshot();var draftBefore=draft.snapshot()) {
            try {
                target.beginPrefill(prompt.size());draft.beginPrefill(teachers);
                for(int index=0;index<prompt.size();index++) {
                    int token=prompt.get(index);
                    if(draft.position()==target.position()-1) draftOutput=teacher(token,targetOutput.hyperHidden(),false);
                    else if(draft.position()!=target.position()) throw new IllegalStateException("Shifted teacher prefix differs");
                    targetOutput=targetForward(token,index==prompt.size()-1);logical++;pending=-1;pendingStop=false;
                }
                target.finishPrefill();finishTeacherPrefill=true;
            } catch(IOException|RuntimeException|Error failure) {
                logical=previousLogical;pending=previousPending;pendingStop=previousStop;targetOutput=previousTarget;draftOutput=previousDraft;
                try {target.restore(targetBefore);draft.restore(draftBefore);rollbacks++;} catch(Exception rollback) {failure.addSuppressed(rollback);}throw failure;
            }
        }
    }
    private boolean verifyRound(int bound,Set<Integer> stops,Sampler sampler,IntConsumer callback,List<Integer> result) throws IOException {
        int count=Math.min(3,Math.min(bound-logical,context-target.position()-1));
        int[] candidates=new int[count];var verified=new ArrayList<Qwen4CpuEngine.Output>();var prefixes=new ArrayList<Checkpoint>();
        int restored=-1;boolean committed=false;int previousPending=pending,previousLogical=logical;boolean previousStop=pendingStop;
        var previousTarget=targetOutput;var previousDraft=draftOutput;
        try(var base=target.snapshot();var draftBase=draft.snapshot()) {
            try {
                var current=draftOutput;
                for(int i=0;i<count;i++) {
                    candidates[i]=current.greedy();proposed++;
                    if(i+1<count) {current=draft.propose(candidates[i],current.hyperHidden());draftForwards++;}
                }
                draft.restore(draftBase);
                verified.add(targetForward(pending,true));prefixes.add(target.snapshot());
                for(int candidate:candidates) {verified.add(targetForward(candidate,true));prefixes.add(target.snapshot());}
                boolean stop=false;
                for(int i=0;i<verified.size() && logical<bound;i++) {
                    int token=sample(verified.get(i),sampler);boolean match=i<count && token==candidates[i];
                    if(match) accepted++;else if(i<count) corrections++;
                    refresh.add(new Feed(token,verified.get(i).hyperHidden()));
                    restored=i;committed=true;targetOutput=verified.get(i);
                    stop=emit(token,stops,result,callback);
                    if(stop || !match) break;
                }
                target.restore(prefixes.get(restored));rounds++;return stop;
            } catch(IOException|RuntimeException|Error failure) {
                try {
                    if(committed) target.restore(prefixes.get(restored));
                    else {target.restore(base);draft.restore(draftBase);logical=previousLogical;pending=previousPending;pendingStop=previousStop;targetOutput=previousTarget;draftOutput=previousDraft;refresh.clear();}
                    rollbacks++;
                } catch(Exception rollback) {failure.addSuppressed(rollback);}throw failure;
            } finally {for(var saved:prefixes) saved.close();}
        }
    }
    @Override public void reset() {
        check();target.reset();draft.reset();refresh.clear();logical=0;pending=-1;pendingStop=finishTeacherPrefill=false;targetOutput=draftOutput=null;
    }
    @Override public void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        try {draft.close();} catch(IOException error) {failure=error;}
        try {target.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        refresh.clear();if(onClose!=null) onClose.run();if(failure!=null) throw failure;
    }
    public static Qwen4SpeculativeSequence create(Qwen4CpuEngine engine,Qwen4MtpBlock mtp,Runnable onClose) throws IOException {
        var sequence=engine.newSequence();Qwen4MtpBlock.State state;
        try {state=mtp.newState();} catch(RuntimeException|Error failure) {try {sequence.close();} catch(IOException closing) {failure.addSuppressed(closing);}throw failure;}
        record TargetSaved(Qwen4CpuEngine.Sequence.Snapshot value) implements Checkpoint {public void close() {value.close();}}
        record DraftSaved(Qwen4MtpBlock.State.Snapshot value) implements Checkpoint {public void close() {value.close();}}
        Target target=new Target() {
            public int position() {return sequence.position();}public Checkpoint snapshot() throws IOException {return new TargetSaved(sequence.snapshot());}
            public void restore(Checkpoint saved) throws IOException {sequence.restore(((TargetSaved)saved).value());}
            public void reset() {sequence.reset();}public void beginPrefill(int n) {sequence.beginPrefill(n);}public void finishPrefill() {sequence.finishPrefill();}
            public Qwen4CpuEngine.Output forward(int token,boolean logits) throws IOException {return engine.forward(token,sequence,logits,null);}public void close() throws IOException {sequence.close();}
        };
        Draft draft=new Draft() {
            public int position() {return state.position();}public Checkpoint snapshot() throws IOException {return new DraftSaved(state.snapshot());}
            public void restore(Checkpoint saved) throws IOException {state.restore(((DraftSaved)saved).value());}
            public void reset() {state.reset();}public void beginPrefill(int n) {state.beginPrefill(n);}public void finishPrefill() {state.finishPrefill();}
            public Qwen4CpuEngine.Output teacher(int token,float[] hidden,boolean logits) throws IOException {return mtp.forward(token,hidden,state,logits,null);}
            public Qwen4CpuEngine.Output propose(int token,float[] hidden) throws IOException {return mtp.forwardDraft(token,hidden,state,true,null);}public void close() throws IOException {state.close();}
        };
        return new Qwen4SpeculativeSequence(target,draft,engine.configuration().contextLength(),engine.tokenizer().tokenCount(),onClose);
    }
}

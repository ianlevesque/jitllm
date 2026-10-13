package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.*;
import org.beehive.jitllm.inference.op.QsaExecutor;
import org.beehive.jitllm.format.safetensors.FloatFormats;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextQsaKernels;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Device-paged physical FP8 KV, BF16 compressed keys, streamed top-512 block selection and sparse GQA.
 * Plans reserve a bounded full-context address space but only execute/allocate device pages as reached.
 * Small pending index-projection inputs are checkpointed on the host; KV never downloads to it.
 */
public final class FlashNextQsaExecutor implements QsaExecutor,AutoCloseable {
    public record Statistics(long reservedBytes,long materializedPageBytes,int sessions,int pages,int liveSnapshots,
                             long forwards,long restores,long kernelNanos,long kvHostDownloadBytes,long indexHostDownloadBytes) {}
    private final TornadoDevice device;
    private final long budget;
    private final int checkpointCapacity;
    private final boolean nativeDecode;
    private final Set<DeviceSession> sessions=Collections.newSetFromMap(new IdentityHashMap<>());
    private long reserved,graphId,forwards,restores,kernelNanos,indexDownloads;
    private boolean closed;
    public FlashNextQsaExecutor(int deviceIndex,int checkpointCapacity,long budget) {
        this(deviceIndex,checkpointCapacity,budget,false);
    }
    public FlashNextQsaExecutor(int deviceIndex,int checkpointCapacity,long budget,boolean nativeDecode) {
        if(checkpointCapacity<=0 || budget<=0) throw new IllegalArgumentException("Positive QSA budget and checkpoint capacity required");
        device=TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);
        this.budget=budget;this.checkpointCapacity=checkpointCapacity;this.nativeDecode=nativeDecode;
    }
    public synchronized Statistics statistics() {
        long materialized=0;int pages=0,snapshots=0;
        for(var session:sessions) {
            snapshots+=session.snapshots.size();
            for(var page:session.pages) if(page.touched) {pages++;materialized+=page.bytes;}
        }
        return new Statistics(reserved,materialized,sessions.size(),pages,snapshots,forwards,restores,kernelNanos,0,indexDownloads);
    }
    @Override public synchronized Session newSession(Shape shape,float[] qNorm,float[] kNorm,float[] iqNorm,float[] ikNorm,int[] axes,float[] frequencies,boolean fp8,boolean fused) {
        if(closed) throw new IllegalStateException("QSA executor is closed");
        if(qNorm.length!=shape.dimension() || kNorm.length!=shape.dimension() || iqNorm.length!=shape.indexDimension() || ikNorm.length!=shape.indexDimension()
                || axes.length!=shape.rotaryDimensions()/2 || frequencies.length!=axes.length) throw new IllegalArgumentException("Invalid QSA normalization/RoPE weights");
        for(int axis:axes) if(axis<0 || axis>=3) throw new IllegalArgumentException("Invalid QSA RoPE axis");
        if(nativeDecode && (!fused || shape.dimension()!=256 || shape.heads()!=24 || shape.kvHeads()!=2 || shape.selectedCapacity()>2112))
            throw new IllegalArgumentException("Native XQA requires the Flash-Next fused page geometry");
        var session=new DeviceSession(shape,qNorm,kNorm,iqNorm,ikNorm,axes,frequencies,fp8,fused);
        sessions.add(session);reserved+=session.bytes;return session;
    }
    private final class DeviceSession implements Session {
        final Shape shape;
        final boolean fp8,fused;
        final Arena arena=Arena.ofShared();
        final List<ImmutableTaskGraph> graphs=new ArrayList<>();
        final Set<Saved> snapshots=Collections.newSetFromMap(new IdentityHashMap<>());
        final float[] pendingHost;
        final int[][] pendingPositions;
        final FloatArray projected,keyInput,valueInput,indexInput,pending,query,key,gate,indexQuery,pooled,normalizedPooled,output,table,scores,probabilities,alpha,denominator,attended;
        final Map<String,FloatArray> traceArrays=new LinkedHashMap<>();
        final IntArray positions,firstPosition,control,selected,diagnostic,frozenSelection;
        final Page[] pages;
        final GridScheduler scheduler=new GridScheduler();
        final int selectionGraph,attentionGraph,diagnosticGraph,frozenGraph,traceGraph;
        final TornadoExecutionPlan execution;
        long bytes,generation;
        int tokens,indexedThrough;
        int[] lastSelection=new int[0];
        boolean freshActive,released;
        final class Page {
            final int start,rows,appendGraph,scoreGraph,gatherGraph;
            final long bytes;
            final ByteArray cache;
            final BFloat16Array compressed,fresh;
            final IntArray appendControl,scoreControl;
            boolean touched;
            Page(int number,String producer,FloatArray bestScores,IntArray bestIndices,BFloat16Array gathered) {
                start=number*shape.pageTokens();rows=Math.min(shape.pageTokens(),shape.context()-start);
                long before=DeviceSession.this.bytes;
                cache=bytes(Math.multiplyExact(Math.multiplyExact(rows,2*shape.kvWidth()),fp8 ? 1 : 2));
                compressed=bfloats(Math.multiplyExact(512,shape.indexDimension()));
                fresh=bfloats(start<Math.min(4096,shape.context()) ? Math.multiplyExact(rows,2*shape.kvWidth()) : 1);
                appendControl=ints(3);scoreControl=ints(2);bytes=DeviceSession.this.bytes-before;
                String appendName=producer+"Page"+number;
                appendGraph=add(new TaskGraph(appendName)
                        .consumeFromDevice(producer,key,valueInput,normalizedPooled,table)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION,cache,compressed,fresh)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,appendControl)
                        .task("append",FlashNextQsaKernels::append,key,valueInput,normalizedPooled,table,cache,compressed,fresh,appendControl,shape.kvWidth(),rows,shape.indexDimension(),shape.ratio(),fp8 ? 1 : 0)
                        .persistOnDevice(cache,compressed,fresh));
                String scoreName=producer+"Score"+number;
                scoreGraph=add(new TaskGraph(scoreName).consumeFromDevice(producer,indexQuery,bestScores,bestIndices)
                        .consumeFromDevice(appendName,compressed)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,scoreControl)
                        .task("merge",FlashNextQsaKernels::mergeTop,new KernelContext(),indexQuery,compressed,bestScores,bestIndices,scoreControl,shape.indexDimension(),shape.indexHeads()));
                var worker=new WorkerGrid1D(1024);worker.setLocalWork(1024,1,1);scheduler.addWorkerGrid(scoreName+".merge",worker);
                gatherGraph=add(new TaskGraph(producer+"Gather"+number).consumeFromDevice(producer,table,selected,gathered,control)
                        .consumeFromDevice(appendName,cache,fresh)
                        .task("gather",FlashNextQsaKernels::gather,cache,fresh,table,selected,gathered,control,shape.kvWidth(),start,rows,fp8 ? 1 : 0));
            }
        }
        DeviceSession(Shape shape,float[] qNorm,float[] kNorm,float[] iqNorm,float[] ikNorm,int[] axes,float[] frequencies,boolean fp8,boolean fused) {
            this.shape=shape;this.fp8=fp8;this.fused=fused;
            pendingHost=new float[shape.ratio()*shape.indexDimension()];pendingPositions=new int[shape.ratio()][3];
            try {
                projected=floats(2*shape.queryWidth());keyInput=floats(shape.kvWidth());valueInput=floats(shape.kvWidth());
                indexInput=floats(shape.indexQueryWidth()+shape.indexDimension());pending=floats(pendingHost.length);
                positions=ints(3);firstPosition=ints(3);control=ints(6);
                query=floats(shape.queryWidth());key=floats(shape.kvWidth());gate=floats(shape.queryWidth());
                indexQuery=floats(shape.indexQueryWidth());pooled=floats(shape.indexDimension());normalizedPooled=floats(shape.indexDimension());
                output=floats(shape.queryWidth());attended=floats(shape.queryWidth());table=floats(256);for(int i=0;i<256;i++) table.set(i,FloatFormats.e4m3(i));
                FloatArray qw=floats(qNorm),kw=floats(kNorm),iqw=floats(iqNorm),ikw=floats(ikNorm),frequency=floats(frequencies);
                IntArray axis=ints(axes),bestIndices=ints(512);var bestScores=floats(512);
                selected=ints(shape.selectedCapacity());diagnostic=ints(selected.getSize());
                frozenSelection=ints(selected.getSize());
                var gathered=bfloats(Math.multiplyExact(shape.selectedCapacity(),2*shape.kvWidth()));
                String producer="flashnextQsa"+(graphId++);
                add(new TaskGraph(producer).transferToDevice(DataTransferMode.FIRST_EXECUTION,qw,kw,iqw,ikw,frequency,axis,table,bestScores,bestIndices,selected,gathered)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,projected,keyInput,valueInput,indexInput,pending,positions,firstPosition,control)
                        .task("qkv",FlashNextQsaKernels::prepareQkv,projected,keyInput,qw,kw,positions,axis,frequency,query,key,gate,shape.dimension(),shape.epsilon(),fused ? 1 : 0)
                        .task("index",FlashNextQsaKernels::prepareIndexer,indexInput,pending,positions,iqw,axis,frequency,indexQuery,pooled,shape.indexDimension(),shape.ratio(),shape.epsilon(),fused ? 1 : 0,control)
                        .task("pool",FlashNextQsaKernels::normalizePooled,pooled,ikw,normalizedPooled,firstPosition,axis,frequency,shape.indexDimension(),shape.epsilon(),fused ? 1 : 0,control)
                        .task("clearTop",FlashNextQsaKernels::clearTop,bestScores,bestIndices)
                        .task("initShared",FlashNextQsaKernels::initializeShared,valueInput,table,control,selected,gathered)
                        .persistOnDevice(query,key,gate,valueInput,indexQuery,normalizedPooled,table,bestScores,bestIndices,selected,gathered,control));
                pages=new Page[(shape.context()+shape.pageTokens()-1)/shape.pageTokens()];
                for(int i=0;i<pages.length;i++) pages[i]=new Page(i,producer,bestScores,bestIndices,gathered);
                selectionGraph=add(new TaskGraph(producer+"Select").consumeFromDevice(producer,bestIndices,selected,control)
                        .task("select",FlashNextQsaKernels::selection,bestIndices,selected,control,shape.ratio(),shape.budget()));
                frozenGraph=add(new TaskGraph(producer+"Frozen").consumeFromDevice(producer,selected)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION,frozenSelection)
                        .task("select",FlashNextQsaKernels::frozenSelection,frozenSelection,selected));
                int tileCapacity=(shape.selectedCapacity()+15)/16;
                scores=floats(shape.heads()*shape.selectedCapacity());probabilities=floats(scores.getSize());alpha=floats(shape.heads()*tileCapacity);denominator=floats(shape.heads());
                String attentionName=producer+"Attention";
                var attentionTasks=new TaskGraph(attentionName).consumeFromDevice(producer,query,gate,gathered,control)
                        .task("scores",FlashNextQsaKernels::attentionScores,query,gathered,scores,control,shape.dimension(),shape.kvHeads(),shape.heads(),shape.selectedCapacity(),fused ? 1 : 0);
                if(fused && shape.dimension()%16==0 && shape.heads()/shape.kvHeads()<=16) {
                    int capacity=(shape.selectedCapacity()+31)/32*32;
                    var packedKeys=bfloats(shape.kvHeads()*capacity*shape.dimension());var packedQueries=bfloats(shape.kvHeads()*16*shape.dimension());
                    var gemmScores=floats(shape.kvHeads()*16*capacity);var alphaBeta=floats(2);alphaBeta.set(0,1);
                    reserve(FlashNextBf16FloatLibrary.WORKSPACE_BYTES);
                    attentionTasks.transferToDevice(DataTransferMode.FIRST_EXECUTION,packedKeys,packedQueries,gemmScores,alphaBeta)
                            .task("packKeys",FlashNextQsaKernels::packScoreKeys,gathered,packedKeys,control,capacity,shape.dimension(),shape.kvHeads())
                            .task("packQueries",FlashNextQsaKernels::packScoreQueries,query,packedQueries,control,shape.dimension(),shape.kvHeads())
                            .libraryTask("scoresGemm",FlashNextBf16FloatLibrary::gqaScores,capacity,shape.dimension(),shape.kvHeads(),control,packedKeys,packedQueries,alphaBeta,gemmScores)
                            .task("readScores",FlashNextQsaKernels::readScoreGemm,gemmScores,scores,control,shape.selectedCapacity(),capacity,shape.heads(),shape.kvHeads(),shape.dimension());
                }
                attentionTasks
                        .task("probabilities",FlashNextQsaKernels::probabilities,scores,probabilities,alpha,denominator,control,shape.heads(),shape.selectedCapacity(),tileCapacity,fused ? 1 : 0)
                        .task("attend",FlashNextQsaKernels::attend,gathered,probabilities,alpha,denominator,gate,output,control,shape.dimension(),shape.kvHeads(),shape.heads(),shape.selectedCapacity(),tileCapacity,fused ? 1 : 0,attended);
                if(nativeDecode) {
                    int capacity=(shape.selectedCapacity()+63)/64*64;
                    reserve(FlashNextXqaLibrary.WORKSPACE_BYTES);
                    var xqaQuery=bfloats(shape.queryWidth());var xqaKeys=bfloats(capacity*shape.kvWidth());var xqaValues=bfloats(xqaKeys.getSize());var xqaOutput=bfloats(shape.queryWidth());
                    var pageTable=ints(capacity/64);for(int i=0;i<pageTable.getSize();i++) pageTable.set(i,i);
                    attentionTasks.transferToDevice(DataTransferMode.FIRST_EXECUTION,xqaQuery,xqaKeys,xqaValues,xqaOutput,pageTable)
                            .task("packXqa",FlashNextQsaKernels::packXqa,query,gathered,control,xqaQuery,xqaKeys,xqaValues,shape.kvWidth())
                            .libraryTask("decodeXqa",FlashNextXqaLibrary::decode,capacity,shape.dimension(),shape.heads(),shape.kvHeads(),control,xqaQuery,xqaKeys,xqaValues,xqaOutput,pageTable)
                            .task("readXqa",FlashNextQsaKernels::readXqa,xqaOutput,gate,output,attended,control);
                }
                attentionGraph=add(attentionTasks
                        .persistOnDevice(scores,probabilities,attended).transferToHost(DataTransferMode.EVERY_EXECUTION,output));
                diagnosticGraph=add(new TaskGraph(producer+"SelectionRead").consumeFromDevice(producer,selected)
                        .task("copy",FlashNextQsaKernels::copySelection,selected,diagnostic).transferToHost(DataTransferMode.EVERY_EXECUTION,diagnostic));
                var trace=new TaskGraph(producer+"Trace").consumeFromDevice(producer,query,key,gate).consumeFromDevice(producer+"Attention",scores,probabilities,attended);
                for(var pair:List.of(Map.entry("normalized_query",query),Map.entry("normalized_key",key),Map.entry("gate",gate),Map.entry("attended",attended),Map.entry("scores",scores),Map.entry("probabilities",probabilities))) {
                    var target=floats(pair.getValue().getSize());traceArrays.put(pair.getKey(),target);
                    trace.task(pair.getKey(),org.beehive.jitllm.backend.tornado.kernels.FlashNextMoeKernels::readWeights,pair.getValue(),target);
                }
                traceGraph=add(trace.transferToHost(DataTransferMode.EVERY_EXECUTION,(Object[])traceArrays.values().toArray(FloatArray[]::new)));
                execution=new TornadoExecutionPlan(graphs.toArray(ImmutableTaskGraph[]::new)).withDevice(device).withProfiler(ProfilerMode.SILENT).withGridScheduler(scheduler);
            } catch(RuntimeException|Error failure) {arena.close();throw failure;}
        }
        void reserve(long count) {bytes=Math.addExact(bytes,count);if(bytes>budget-reserved) throw new IllegalStateException("QSA context reservation exceeds budget "+budget);}
        FloatArray floats(int count) {long n=TornadoNativeArray.ARRAY_HEADER+4L*count;reserve(n);return FloatArray.fromSegmentShallow(arena.allocate(n,32));}
        FloatArray floats(float[] values) {var out=floats(values.length);put(out,values);return out;}
        IntArray ints(int count) {long n=TornadoNativeArray.ARRAY_HEADER+4L*count;reserve(n);return IntArray.fromSegmentShallow(arena.allocate(n,32));}
        IntArray ints(int[] values) {var out=ints(values.length);for(int i=0;i<values.length;i++) out.set(i,values[i]);return out;}
        ByteArray bytes(int count) {long n=TornadoNativeArray.ARRAY_HEADER+(long)count;reserve(n);return ByteArray.fromSegmentShallow(arena.allocate(n,32));}
        BFloat16Array bfloats(int count) {long n=TornadoNativeArray.ARRAY_HEADER+2L*count;reserve(n);return BFloat16Array.fromSegmentShallow(arena.allocate(n,32));}
        int add(TaskGraph graph) {int index=graphs.size();graphs.add(graph.snapshot());return index;}
        void requireOpen() {if(released || closed) throw new IllegalStateException("QSA session is closed");}
        TornadoExecutionResult run(int graph) throws IOException {
            requireOpen();
            try {var result=execution.withGraph(graph).execute();long nanos=result.getProfilerResult().getDeviceKernelTime();if(nanos<=0) throw new IOException("QSA graph recorded no CUDA kernel");kernelNanos+=nanos;return result;}
            catch(RuntimeException failure) {throw new IOException("GPU QSA graph "+graph+" failed",failure);}
        }
        @Override public int tokens() {return tokens;}
        @Override public float[] forward(float[] qg,float[] k,float[] v,float[] index,int[] coordinates,boolean prefill,boolean fresh,int blockSize) throws IOException {
            return forward(qg,k,v,index,coordinates,prefill,fresh,blockSize,null);
        }
        @Override public float[] forwardFrozen(float[] qg,float[] k,float[] v,int[] coordinates,int[] selection) throws IOException {
            return forward(qg,k,v,new float[indexInput.getSize()],coordinates,false,false,32,selection);
        }
        private float[] forward(float[] qg,float[] k,float[] v,float[] index,int[] coordinates,boolean prefill,boolean fresh,int blockSize,int[] frozen) throws IOException {
            synchronized(FlashNextQsaExecutor.this) {
                requireOpen();
                if(frozen==null && indexedThrough!=tokens) throw new IllegalArgumentException("Restore indexed prefix before draft extend");
                if(frozen!=null) {
                    if(shape.draftTail()==0 || frozen.length==0 || frozen.length>selected.getSize()) throw new IllegalArgumentException("Invalid MTP sparse capacity");
                    var seen=new HashSet<Integer>();for(int token:frozen) if(token<0 || token>tokens || !seen.add(token)) throw new IllegalArgumentException("Invalid MTP sparse token");
                }
                if(tokens>=shape.context() || coordinates.length!=3 || (fresh && (!prefill || tokens>=Math.min(4096,shape.context()) || (tokens>0 && !freshActive)))
                        || (blockSize!=16 && blockSize!=32 && blockSize!=64)) throw new IllegalArgumentException("Invalid QSA mode/context");
                for(int coordinate:coordinates) if(coordinate<0) throw new IllegalArgumentException("Negative QSA RoPE coordinate");
                put(projected,qg);put(keyInput,k);put(valueInput,v);put(indexInput,index);
                int slot=tokens%shape.ratio();if(frozen==null) {System.arraycopy(index,shape.indexQueryWidth(),pendingHost,slot*shape.indexDimension(),shape.indexDimension());pendingPositions[slot]=coordinates.clone();}
                put(pending,pendingHost);for(int i=0;i<3;i++) {positions.set(i,coordinates[i]);firstPosition.set(i,pendingPositions[0][i]);}
                int visible=tokens+1,count=frozen==null ? Math.min(visible/shape.ratio(),shape.budget()/shape.ratio())*shape.ratio()+visible%shape.ratio() : frozen.length;
                control.set(0,visible);control.set(1,fresh ? 1 : 0);control.set(2,count);control.set(3,prefill ? 1 : 0);control.set(4,blockSize);
                control.set(5,frozen==null ? 0 : 1);
                run(0);var current=pages[tokens/shape.pageTokens()];
                current.appendControl.set(0,tokens-current.start);current.appendControl.set(1,fresh ? 1 : 0);current.appendControl.set(2,frozen==null && visible%shape.ratio()==0 ? 1 : 0);
                run(current.appendGraph);current.touched=true;
                int blocks=visible/shape.ratio();
                if(frozen==null) {for(var page:pages) {
                    int visibleBlocks=Math.min(512,blocks-page.start/shape.ratio());if(visibleBlocks<=0) break;
                    page.scoreControl.set(0,visibleBlocks);page.scoreControl.set(1,page.start/shape.ratio());run(page.scoreGraph);
                }
                run(selectionGraph);
                } else {
                    for(int i=0;i<frozenSelection.getSize();i++) frozenSelection.set(i,i<frozen.length ? frozen[i] : -1);
                    run(frozenGraph);
                }
                for(int page=0;page<=tokens/shape.pageTokens();page++) run(pages[page].gatherGraph);
                run(attentionGraph);
                if(frozen==null) {run(diagnosticGraph);lastSelection=new int[count];for(int i=0;i<count;i++) lastSelection[i]=diagnostic.get(i);indexDownloads+=4L*diagnostic.getSize();}
                else lastSelection=frozen.clone();
                float[] result=new float[output.getSize()];for(int i=0;i<result.length;i++) result[i]=output.get(i);
                tokens=visible;if(frozen==null) indexedThrough=tokens;freshActive=fresh;forwards++;return result;
            }
        }
        @Override public Map<String,float[]> intermediates() throws IOException {
            synchronized(FlashNextQsaExecutor.this) {
                requireOpen();if(tokens==0) throw new IllegalStateException("No QSA forward to trace");run(traceGraph);
                var result=new LinkedHashMap<String,float[]>();int count=lastSelection.length;
                for(var pair:traceArrays.entrySet()) {
                    boolean compact=pair.getKey().equals("scores") || pair.getKey().equals("probabilities");
                    float[] row=new float[compact ? shape.heads()*count : pair.getValue().getSize()];
                    for(int i=0;i<row.length;i++) row[i]=pair.getValue().get(compact ? i/count*shape.selectedCapacity()+i%count : i);
                    String name=nativeDecode && control.get(3)==0 && compact ? "reference_"+pair.getKey() : pair.getKey();result.put(name,row);
                }return result;
            }
        }
        private final class Saved implements Snapshot {
            final DeviceSession owner=DeviceSession.this;
            final long epoch=generation;
            final int length=tokens;
            final int indexed=indexedThrough;
            final float[] pending=pendingHost.clone();
            final int[][] positions=Arrays.stream(pendingPositions).map(int[]::clone).toArray(int[][]::new);
            final int[] selection=lastSelection.clone();
            final boolean fresh=freshActive;
            boolean disposed,valid=true;
            @Override public void close() {synchronized(FlashNextQsaExecutor.this) {if(!disposed) {disposed=true;snapshots.remove(this);}}}
        }
        @Override public Snapshot snapshot() {
            synchronized(FlashNextQsaExecutor.this) {requireOpen();if(snapshots.size()>=checkpointCapacity) throw new IllegalStateException("QSA checkpoint capacity exhausted");var saved=new Saved();snapshots.add(saved);return saved;}
        }
        @Override public void restore(Snapshot snapshot) {
            synchronized(FlashNextQsaExecutor.this) {
                requireOpen();if(!(snapshot instanceof Saved saved) || saved.owner!=this || saved.epoch!=generation || saved.disposed || !saved.valid || saved.length>tokens)
                    throw new IllegalArgumentException("Invalid QSA checkpoint owner/lifetime/prefix");
                for(var future:snapshots) if(future.length>saved.length) future.valid=false;
                tokens=saved.length;indexedThrough=saved.indexed;System.arraycopy(saved.pending,0,pendingHost,0,pendingHost.length);
                for(int i=0;i<pendingPositions.length;i++) pendingPositions[i]=saved.positions[i].clone();
                lastSelection=saved.selection.clone();freshActive=saved.fresh;restores++;
            }
        }
        @Override public int[] selectedTokens() {synchronized(FlashNextQsaExecutor.this) {requireOpen();return lastSelection.clone();}}
        @Override public void finishFreshPrefill() {synchronized(FlashNextQsaExecutor.this) {requireOpen();freshActive=false;}}
        @Override public void reset() {
            synchronized(FlashNextQsaExecutor.this) {
                requireOpen();tokens=0;indexedThrough=0;generation++;snapshots.clear();freshActive=false;Arrays.fill(pendingHost,0);
                for(var position:pendingPositions) Arrays.fill(position,0);lastSelection=new int[0];
            }
        }
        @Override public void close() throws IOException {
            synchronized(FlashNextQsaExecutor.this) {
                if(released) return;released=true;generation++;snapshots.clear();
                try {execution.close();} catch(Exception failure) {throw new IOException("Cannot release QSA plans",failure);}
                finally {arena.close();if(sessions.remove(this)) reserved-=bytes;}
            }
        }
    }
    private static void put(FloatArray out,float[] values) {if(values.length!=out.getSize()) throw new IllegalArgumentException("Wrong QSA projection width");for(int i=0;i<values.length;i++) out.set(i,values[i]);}
    @Override public synchronized void close() throws IOException {
        if(closed) return;IOException failure=null;
        for(var session:List.copyOf(sessions)) try {session.close();} catch(IOException closing) {if(failure==null) failure=closing;else failure.addSuppressed(closing);}
        closed=true;if(failure!=null) throw failure;
    }
}

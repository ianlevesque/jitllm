package org.beehive.jitllm.backend.tornado;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.beehive.jitllm.format.safetensors.*;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.*;

/** Packed FP8 expert residency independent of compute graphs, with leases that prevent in-flight eviction. */
final class FlashNextFp8ExpertCache implements AutoCloseable {
    private static final AtomicInteger NEXT_ID=new AtomicInteger(1);
    record Key(ModelOptMatrix first,ModelOptMatrix second) {}
    record Resource(int id,float[] scales) {}
    final TornadoDevice device;
    long budget;
    long resident,uploads,evictions,readBytes,hits,misses,evictedBytes;
    final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    boolean closed;
    FlashNextFp8ExpertCache(TornadoDevice device,long budget) {this.device=device;this.budget=budget;}
    final class Lease implements AutoCloseable {
        final Set<Key> keys=new HashSet<>();
        Resource get(ModelOptMatrix first,ModelOptMatrix second) throws IOException {
            var key=new Key(first,second);var entry=cache.get(key);
            if(entry!=null) hits++;
            if(entry==null) {
                misses++;
                long size=size(key);if(size>budget) throw new IOException("Expert resource exceeds budget");
                while(size>budget-resident) {
                    var candidate=cache.entrySet().iterator();Entry victim=null;
                    while(candidate.hasNext()) {var item=candidate.next();if(!keys.contains(item.getKey())) {victim=item.getValue();candidate.remove();break;}}
                    if(victim==null) throw new IOException("Expert cache cannot hold the pinned selected experts");
                    victim.close();resident-=victim.bytes;evictions++;evictedBytes+=victim.bytes;
                }
                entry=new Entry(key);cache.put(key,entry);resident+=entry.bytes;uploads++;
            }
            keys.add(key);return new Resource(entry.id,entry.scales);
        }
        @Override public void close() {keys.clear();}
    }
    Lease lease() {if(closed) throw new IllegalStateException("Expert cache is closed");return new Lease();}
    private static long size(Key key) {return key.first.weights().bytes()+4L*key.first.blockScales().elements()+(key.second==null ? 0 : key.second.weights().bytes()+4L*key.second.blockScales().elements())+TornadoNativeArray.ARRAY_HEADER;}
    final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();final TornadoExecutionPlan plan;final long bytes;final int id;final float[] scales;
        Entry(Key key) throws IOException {
            bytes=size(key);id=NEXT_ID.getAndIncrement();if(id<=0) throw new IllegalStateException("Expert resource ID exhausted");
            TornadoExecutionPlan built=null;
            try {
                var first=key.first;var second=key.second;int rows=first.rows()+(second==null ? 0 : second.rows()),columns=first.columns();
                var weights=ByteArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+(long)rows*columns,32));
                scales=new float[rows/128*(columns/128)];
                load(first,weights,scales,0);if(second!=null) load(second,weights,scales,first.rows());
                var graph=new TaskGraph("flashnextFp8ExpertResource"+id).transferToDevice(DataTransferMode.FIRST_EXECUTION,weights)
                        .libraryTask("register",FlashNextFp8Library::register,id,rows,columns,weights);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device);built.execute();plan=built;
            } catch(Exception|Error error) {if(built!=null) try {built.close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}arena.close();throw new IOException("Cannot stage packed expert",error);}
        }
        void load(ModelOptMatrix matrix,ByteArray weights,float[] scales,int rowOffset) throws IOException {
            var tensor=matrix.weights();
            try(var temporary=Arena.ofConfined();var file=FileChannel.open(tensor.file(),StandardOpenOption.READ)) {
                weights.getSegment().asSlice((long)rowOffset*matrix.columns(),tensor.bytes()).copyFrom(file.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),temporary));
            }
            tensor=matrix.blockScales();byte[] raw=new byte[Math.toIntExact(tensor.bytes())];
            try(var file=FileChannel.open(tensor.file(),StandardOpenOption.READ)) {SafeTensorIndex.readFully(file,java.nio.ByteBuffer.wrap(raw),tensor.offset());}
            var buffer=java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            int offset=rowOffset/128*(matrix.columns()/128);
            for(int i=0;i<tensor.elements();i++) scales[offset+i]=tensor.dtype().equals("F32") ? buffer.getFloat() : org.beehive.jitllm.format.safetensors.FloatFormats.bf16(buffer.getShort());
            readBytes+=matrix.weights().bytes()+matrix.blockScales().bytes();
        }
        @Override public void close() throws IOException {try {plan.close();} catch(Exception error) {throw new IOException("Cannot release expert registration",error);} finally {arena.close();}}
    }
    @Override public void close() throws IOException {
        if(closed) return;closed=true;IOException failure=null;
        for(var entry:cache.values()) try {entry.close();} catch(IOException error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
        cache.clear();resident=0;if(failure!=null) throw failure;
    }
}

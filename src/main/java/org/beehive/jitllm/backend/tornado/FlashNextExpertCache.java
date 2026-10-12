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

/** Packed expert residency independent of compute graphs, with leases that prevent in-flight eviction. */
final class FlashNextExpertCache implements AutoCloseable {
    private static final AtomicInteger NEXT_ID=new AtomicInteger(1);
    record Key(ModelOptMatrix first,ModelOptMatrix second) {}
    record Resource(int id,float scale) {}
    final TornadoDevice device;
    long budget;
    long resident,uploads,evictions,readBytes;
    final LinkedHashMap<Key,Entry> cache=new LinkedHashMap<>(16,.75f,true);
    boolean closed;
    FlashNextExpertCache(TornadoDevice device,long budget) {this.device=device;this.budget=budget;}
    final class Lease implements AutoCloseable {
        final Set<Key> keys=new HashSet<>();
        Resource get(ModelOptMatrix first,ModelOptMatrix second) throws IOException {
            var key=new Key(first,second);var entry=cache.get(key);
            if(entry==null) {
                long size=size(key);if(size>budget) throw new IOException("Expert resource exceeds budget");
                while(size>budget-resident) {
                    var candidate=cache.entrySet().iterator();Entry victim=null;
                    while(candidate.hasNext()) {var item=candidate.next();if(!keys.contains(item.getKey())) {victim=item.getValue();candidate.remove();break;}}
                    if(victim==null) throw new IOException("Expert cache cannot hold the pinned selected experts");
                    victim.close();resident-=victim.bytes;evictions++;
                }
                entry=new Entry(key);cache.put(key,entry);resident+=entry.bytes;uploads++;
            }
            keys.add(key);return new Resource(entry.id,first.globalScale());
        }
        @Override public void close() {keys.clear();}
    }
    Lease lease() {if(closed) throw new IllegalStateException("Expert cache is closed");return new Lease();}
    private static long size(Key key) {return key.first.weights().bytes()+key.first.blockScales().bytes()+(key.second==null ? 0 : key.second.weights().bytes()+key.second.blockScales().bytes())+2L*TornadoNativeArray.ARRAY_HEADER;}
    final class Entry implements AutoCloseable {
        final Arena arena=Arena.ofShared();final TornadoExecutionPlan plan;final long bytes;final int id;
        Entry(Key key) throws IOException {
            bytes=size(key);id=NEXT_ID.getAndIncrement();if(id<=0) throw new IllegalStateException("Expert resource ID exhausted");
            TornadoExecutionPlan built=null;
            try {
                var first=key.first;var second=key.second;int rows=first.rows()+(second==null ? 0 : second.rows()),columns=first.columns();
                var weights=ByteArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+(long)rows*columns/2,32));
                var scales=ByteArray.fromSegmentShallow(arena.allocate(TornadoNativeArray.ARRAY_HEADER+(long)rows*columns/16,32));
                load(first,weights,scales,0);if(second!=null) load(second,weights,scales,first.rows());
                var graph=new TaskGraph("flashnextExpertResource"+id).transferToDevice(DataTransferMode.FIRST_EXECUTION,weights,scales)
                        .libraryTask("register",FlashNextNvfp4Library::register,id,rows,columns,weights,scales);
                built=new TornadoExecutionPlan(graph.snapshot()).withDevice(device);built.execute();plan=built;
            } catch(Exception|Error error) {if(built!=null) try {built.close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}arena.close();throw new IOException("Cannot stage packed expert",error);}
        }
        void load(ModelOptMatrix matrix,ByteArray weights,ByteArray scales,int rowOffset) throws IOException {
            var tensor=matrix.weights();
            try(var temporary=Arena.ofConfined();var file=FileChannel.open(tensor.file(),StandardOpenOption.READ)) {
                weights.getSegment().asSlice((long)rowOffset*matrix.columns()/2,tensor.bytes()).copyFrom(file.map(FileChannel.MapMode.READ_ONLY,tensor.offset(),tensor.bytes(),temporary));
            }
            tensor=matrix.blockScales();byte[] raw=new byte[Math.toIntExact(tensor.bytes())];
            try(var file=FileChannel.open(tensor.file(),StandardOpenOption.READ)) {SafeTensorIndex.readFully(file,java.nio.ByteBuffer.wrap(raw),tensor.offset());}
            int columns=matrix.columns()/16;
            for(int row=0;row<matrix.rows();row++) for(int column=0;column<columns;column++) {
                int targetRow=row+rowOffset;int offset=(targetRow/128*(columns/4)+column/4)*512+(targetRow%32)*16+(targetRow%128/32)*4+column%4;
                scales.set(offset,raw[row*columns+column]);
            }
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

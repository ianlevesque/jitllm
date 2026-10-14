package org.beehive.jitllm.backend.cpu;

import org.junit.Test;
import static org.junit.Assert.*;

public class Qwen4PrefillGeometryTest {
    private static int legacyRows(int position,int prefillStart,int prefillEnd) {
        if(position>=prefillEnd) return 1;
        int chunkStart=position-(position-prefillStart)%4096;
        return Math.min(4096,prefillEnd-chunkStart);
    }

    @Test public void defaultChunkMatchesLegacyGeometryExactly() {
        for(int start:new int[]{0,1,7,4090,8192})
            for(int length:new int[]{1,2,16,17,100,4096,4097,10000}) {
                int end=start+length;
                for(int position=start;position<=end;position++) {
                    int expected=legacyRows(position,start,end);
                    assertEquals(expected,Qwen4CpuEngine.PrefillGeometry.rows(position,start,end,1));
                    assertEquals(expected,Qwen4CpuEngine.PrefillGeometry.rows(position,start,end,4096));
                    assertEquals(expected,Qwen4CpuEngine.PrefillGeometry.rows(position,start,end,8192));
                }
            }
    }

    @Test public void chunkedGeometryCapsAndAlignsToThePrefillStart() {
        assertEquals(4096,Qwen4CpuEngine.PrefillGeometry.effectiveChunk(1));
        assertEquals(32,Qwen4CpuEngine.PrefillGeometry.effectiveChunk(32));
        assertEquals(4096,Qwen4CpuEngine.PrefillGeometry.effectiveChunk(100000));
        // Prefill [10, 200) with chunk 32: chunks start at 10, 42, 74, ...
        assertEquals(32,Qwen4CpuEngine.PrefillGeometry.rows(10,10,200,32));
        assertEquals(32,Qwen4CpuEngine.PrefillGeometry.rows(41,10,200,32));
        assertEquals(32,Qwen4CpuEngine.PrefillGeometry.rows(42,10,200,32));
        assertEquals(30,Qwen4CpuEngine.PrefillGeometry.rows(195,10,200,32));
        assertEquals(10,Qwen4CpuEngine.PrefillGeometry.chunkStart(41,10,32));
        assertEquals(42,Qwen4CpuEngine.PrefillGeometry.chunkStart(42,10,32));
        // A prompt shorter than the chunk reports its own remaining length.
        assertEquals(5,Qwen4CpuEngine.PrefillGeometry.rows(0,0,5,32));
        assertEquals(5,Qwen4CpuEngine.PrefillGeometry.rows(4,0,5,32));
    }

    @Test public void decodeAlwaysReportsOneRow() {
        assertEquals(1,Qwen4CpuEngine.PrefillGeometry.rows(200,10,200,32));
        assertEquals(1,Qwen4CpuEngine.PrefillGeometry.rows(5000,10,200,32));
        assertEquals(1,Qwen4CpuEngine.PrefillGeometry.rows(0,0,0,1));
    }

    @Test public void rowsStayWithinChunkAndPrefillBounds() {
        for(int chunk:new int[]{2,3,16,17,32,64,100,1000,4096})
            for(int start:new int[]{0,3,100})
                for(int length:new int[]{1,31,32,33,500}) {
                    int end=start+length;
                    for(int position=start;position<end;position++) {
                        int rows=Qwen4CpuEngine.PrefillGeometry.rows(position,start,end,chunk);
                        assertTrue(rows>=1 && rows<=Math.min(chunk,length));
                        int chunkStart=Qwen4CpuEngine.PrefillGeometry.chunkStart(position,start,chunk);
                        assertTrue(chunkStart<=position && chunkStart>=start);
                        assertEquals(0,(chunkStart-start)%Qwen4CpuEngine.PrefillGeometry.effectiveChunk(chunk));
                    }
                }
    }

    @Test public void invalidChunksAreRejected() {
        assertThrows(IllegalArgumentException.class,()->Qwen4CpuEngine.PrefillGeometry.effectiveChunk(0));
        assertThrows(IllegalArgumentException.class,()->Qwen4CpuEngine.PrefillGeometry.rows(0,0,10,-1));
    }
}

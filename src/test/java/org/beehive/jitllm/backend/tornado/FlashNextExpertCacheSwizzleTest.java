package org.beehive.jitllm.backend.tornado;

import org.junit.Test;
import static org.junit.Assert.*;

public class FlashNextExpertCacheSwizzleTest {
    @Test public void swizzleIsABijectionOverFullScaleTiles() {
        for(int[] shape:new int[][]{{640,160},{2560,40},{128,4},{128,160},{1024,20}}) {
            int rows=shape[0],columns=shape[1];
            boolean[] seen=new boolean[rows*columns];
            for(int row=0;row<rows;row++) for(int column=0;column<columns;column++) {
                int offset=FlashNextExpertCache.swizzledOffset(row,column,columns);
                assertTrue("offset in range",offset>=0 && offset<seen.length);
                assertFalse("offset unique at "+row+","+column,seen[offset]);
                seen[offset]=true;
            }
        }
    }

    @Test public void swizzleMatchesTheReferenceTileLayout() {
        // Reference formulation as documented for the cuBLASLt 128x4 tile:
        // tiles are ordered row-major over (row/128, column/4); inside a tile the
        // byte index is (row%32)*16 + (row%128/32)*4 + column%4.
        for(int row:new int[]{0,1,31,32,127,128,639,2559})
            for(int column:new int[]{0,3,4,39,159}) {
                int columns=160;
                int expected=(row/128*(columns/4)+column/4)*512+(row%32)*16+(row%128/32)*4+column%4;
                assertEquals(expected,FlashNextExpertCache.swizzledOffset(row,column,columns));
            }
    }
}

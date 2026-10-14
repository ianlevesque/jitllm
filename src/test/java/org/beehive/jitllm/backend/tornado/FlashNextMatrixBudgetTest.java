package org.beehive.jitllm.backend.tornado;

import org.junit.Test;
import static org.junit.Assert.*;

public class FlashNextMatrixBudgetTest {
    private static final long MIB=1L<<20,GIB=1L<<30;

    @Test public void plainAndNvfp4PartitionTheBudgetAsBefore() {
        var split=FlashNextMatrixExecutor.splitBudgets(8*GIB,false,-1,-1);
        assertEquals(8*GIB,split.plain());assertEquals(0,split.nvfp4());
        split=FlashNextMatrixExecutor.splitBudgets(8*GIB,true,-1,-1);
        assertEquals(4*GIB,split.plain());assertEquals(4*GIB,split.nvfp4());
        split=FlashNextMatrixExecutor.splitBudgets(101,true,-1,-1);
        assertEquals(50,split.plain());assertEquals(51,split.nvfp4());
    }

    @Test public void probeScaleBudgetsKeepTheLegacySubCacheSizes() {
        for(long budget:new long[]{1,2,3,1000,MIB,200*MIB,256*MIB}) {
            var split=FlashNextMatrixExecutor.splitBudgets(budget,false,-1,-1);
            long legacyFloat=budget<2 ? 0 : Math.max(1,budget/4);
            assertEquals("float budget "+budget,legacyFloat,split.bf16Float());
            assertEquals("bf16 budget "+budget,budget-legacyFloat,split.bf16());
            assertEquals(budget,split.plain());
        }
    }

    @Test public void multiGigabyteBudgetsScaleSubCachesUpToTheCap() {
        var split=FlashNextMatrixExecutor.splitBudgets(8*GIB,false,-1,-1);
        assertEquals(2*GIB,split.bf16Float());assertEquals(GIB,split.bf16());
        split=FlashNextMatrixExecutor.splitBudgets(4*GIB,false,-1,-1);
        assertEquals(GIB,split.bf16Float());assertEquals(512*MIB,split.bf16());
        split=FlashNextMatrixExecutor.splitBudgets(64*GIB,false,-1,-1);
        assertEquals(2*GIB,split.bf16Float());assertEquals(2*GIB,split.bf16());
        // The shares never shrink below the legacy fixed sizes.
        split=FlashNextMatrixExecutor.splitBudgets(512*MIB,false,-1,-1);
        assertTrue(split.bf16Float()>=64*MIB);assertTrue(split.bf16()>=192*MIB);
    }

    @Test public void overridesWinAndZeroDisablesTheFloatCache() {
        var split=FlashNextMatrixExecutor.splitBudgets(8*GIB,false,12345,6789);
        assertEquals(12345,split.bf16());assertEquals(6789,split.bf16Float());
        assertEquals(0,FlashNextMatrixExecutor.splitBudgets(8*GIB,false,-1,0).bf16Float());
        // A zero BF16 override is ignored rather than disabling a required cache.
        assertEquals(GIB,FlashNextMatrixExecutor.splitBudgets(8*GIB,false,0,-1).bf16());
    }

    @Test public void invalidBudgetsAreRejected() {
        assertThrows(IllegalArgumentException.class,()->FlashNextMatrixExecutor.splitBudgets(0,false,-1,-1));
        assertThrows(IllegalArgumentException.class,()->FlashNextMatrixExecutor.splitBudgets(-1,false,-1,-1));
        assertThrows(IllegalArgumentException.class,()->FlashNextMatrixExecutor.splitBudgets(1,true,-1,-1));
    }
}

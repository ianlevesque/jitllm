package org.beehive.jitllm.backend.cpu;

import static org.junit.Assert.*;
import java.util.*;
import org.junit.Test;

public final class Qwen4SelectionTest {
    @Test public void qsaRanksCompleteBlocksAndKeepsPendingTail() {
        float[] query={1,0,0,1};
        var blocks=List.of(new float[]{1,0},new float[]{-1,2},new float[]{0,-3},new float[]{2,3});
        assertArrayEquals(new int[]{12,13,14,15,4,5,6,7,16},Qwen4QsaAttention.select(query,blocks,17,2,2,8,4));
    }
    @Test public void qsaBeforeFirstBlockIncludesEveryVisibleToken() {
        assertArrayEquals(new int[]{0,1,2},Qwen4QsaAttention.select(new float[]{1,0},List.of(),3,1,2,8,4));
    }
    @Test public void qsaNativeBudgetActuallyExcludesOlderLowScoreBlocks() {
        List<float[]> blocks=new ArrayList<>();for(int i=0;i<600;i++) blocks.add(new float[]{i,i});
        int[] selected=Qwen4QsaAttention.select(new float[]{1,0,0,1},blocks,2403,2,2,2048,4);
        assertEquals(2051,selected.length);
        assertArrayEquals(new int[]{2396,2397,2398,2399},Arrays.copyOf(selected,4));
        assertArrayEquals(new int[]{2400,2401,2402},Arrays.copyOfRange(selected,2048,2051));
        assertEquals(2051,Arrays.stream(selected).distinct().count());
        for(int token:selected) assertTrue(token>=352 && token<2403);
    }
    @Test public void qsaTiesAreDeterministicAndBadLayoutsAreRejected() {
        var blocks=List.of(new float[]{1},new float[]{2},new float[]{3});
        assertArrayEquals(new int[]{0,1,2,3},Qwen4QsaAttention.select(new float[]{0},blocks,6,1,1,4,2));
        assertThrows(IllegalArgumentException.class,()->Qwen4QsaAttention.select(new float[]{1},blocks,8,1,1,4,2));
    }
    @Test public void routerRenormalizesOnlyTheSelectedExperts() {
        float[] logits={0,(float)Math.log(2),(float)Math.log(4)};
        var normalized=Qwen4Moe.route(logits,2,true);
        assertArrayEquals(new int[]{2,1},normalized.experts());
        assertArrayEquals(new float[]{2f/3,1f/3},normalized.weights(),1e-6f);
        var ordinary=Qwen4Moe.route(logits,2,false);
        assertArrayEquals(new float[]{4f/7,2f/7},ordinary.weights(),1e-6f);
    }
    @Test public void routerTiesAndInvalidInputHaveExplicitBehavior() {
        var tied=Qwen4Moe.route(new float[]{0,0,0},2,true);
        assertArrayEquals(new int[]{0,1},tied.experts());
        assertArrayEquals(new float[]{.5f,.5f},tied.weights(),0);
        assertThrows(IllegalArgumentException.class,()->Qwen4Moe.route(new float[]{Float.NaN,0},1,true));
        assertThrows(IllegalArgumentException.class,()->Qwen4Moe.route(new float[]{0},2,true));
    }
}

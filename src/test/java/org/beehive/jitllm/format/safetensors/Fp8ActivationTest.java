package org.beehive.jitllm.format.safetensors;

import static org.junit.Assert.*;
import org.junit.Test;

public final class Fp8ActivationTest {
    @Test public void nearestEvenTiesPreserveSignAndEachGroupsScale() {
        float[] input=new float[256];input[127]=448;input[255]=224;
        input[0]=1.0625f;input[1]=1.1875f;input[2]=-1.0625f;input[3]=-0f;
        input[128]=.53125f;input[129]=.59375f;
        var packed=Fp8Activation.quantize(input);
        assertEquals(0x38,Byte.toUnsignedInt(packed.values()[0]));
        assertEquals(0x3a,Byte.toUnsignedInt(packed.values()[1]));
        assertEquals(0xb8,Byte.toUnsignedInt(packed.values()[2]));
        assertEquals(0x80,Byte.toUnsignedInt(packed.values()[3]));
        assertEquals(packed.values()[0],packed.values()[128]);
        assertEquals(packed.values()[1],packed.values()[129]);
        assertArrayEquals(new float[]{1,.5f},packed.scales(),0);
    }
    @Test public void zeroGroupsStayFiniteAndInvalidInputsFail() {
        var packed=Fp8Activation.quantize(new float[128]);
        assertArrayEquals(new byte[128],packed.values());
        assertTrue(Float.isFinite(packed.scales()[0]) && packed.scales()[0]>0);
        assertThrows(IllegalArgumentException.class,()->Fp8Activation.quantize(new float[127]));
        float[] invalid=new float[128];invalid[4]=Float.NaN;
        assertThrows(IllegalArgumentException.class,()->Fp8Activation.quantize(invalid));
        invalid[4]=Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,()->Fp8Activation.quantize(invalid));
        assertThrows(IllegalArgumentException.class,()->new Fp8Activation.Packed(new byte[128],new float[]{0}));
    }
}

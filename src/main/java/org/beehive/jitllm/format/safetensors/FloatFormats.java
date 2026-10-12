package org.beehive.jitllm.format.safetensors;

/** NVIDIA ModelOpt storage encodings (not GGML block quantization). */
public final class FloatFormats {
    private FloatFormats() {}
    private static final float[] E2M1 = {0f, .5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    public static float e2m1(int nibble) {
        return (nibble & 8) == 0 ? E2M1[nibble & 7] : -E2M1[nibble & 7];
    }

    public static float e4m3(int bits) {
        bits &= 255;
        int exponent = (bits >>> 3) & 15, mantissa = bits & 7;
        float value;
        if (exponent == 15 && mantissa == 7) value = Float.NaN;
        else if (exponent == 0) value = Math.scalb((float) mantissa, -9);
        else value = Math.scalb(1f + mantissa / 8f, exponent - 7);
        return (bits & 128) == 0 ? value : -value;
    }

    public static float bf16(int bits) { return Float.intBitsToFloat((bits & 65535) << 16); }

    public static float roundBf16(float value) {
        if (!Float.isFinite(value)) return value;
        int bits = Float.floatToRawIntBits(value);
        bits += 0x7fff + ((bits >>> 16) & 1);
        return bf16(bits >>> 16);
    }
}

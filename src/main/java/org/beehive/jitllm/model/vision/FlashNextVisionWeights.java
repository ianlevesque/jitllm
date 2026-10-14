package org.beehive.jitllm.model.vision;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.format.safetensors.FloatFormats;
import org.beehive.jitllm.format.safetensors.PagedTensorReader;
import org.beehive.jitllm.format.safetensors.SafeTensorIndex;

/**
 * BF16 vision-tower weights ({@code model.visual.*}) loaded through the existing
 * safetensors index/reader. Matrices are kept as raw BF16 bit patterns (one
 * {@code short} per element, row-major {@code [out][in]}) and expanded to floats
 * per use by the tower, so resident vision weights stay at checkpoint size
 * (about 0.9 GB for this checkpoint) instead of doubling as float32. Vectors
 * (norms, biases, position table) are held as floats. Every shape is validated
 * against {@link FlashNextVisionConfig} at load.
 */
public final class FlashNextVisionWeights {

    /** One transformer block: fused QKV, attention projection, MLP pair, two LayerNorms. */
    public record Block(short[] qkvWeight, float[] qkvBias, short[] projWeight, float[] projBias,
                        short[] fc1Weight, float[] fc1Bias, short[] fc2Weight, float[] fc2Bias,
                        float[] norm1Weight, float[] norm1Bias, float[] norm2Weight, float[] norm2Bias) {}

    private final FlashNextVisionConfig config;
    private final short[] patchWeight;
    private final float[] patchBias;
    private final float[] posEmbed;
    private final List<Block> blocks;
    private final float[] mergerNormWeight;
    private final float[] mergerNormBias;
    private final short[] mergerFc1Weight;
    private final float[] mergerFc1Bias;
    private final short[] mergerFc2Weight;
    private final float[] mergerFc2Bias;

    public FlashNextVisionWeights(FlashNextVisionConfig config, short[] patchWeight, float[] patchBias,
                                  float[] posEmbed, List<Block> blocks, float[] mergerNormWeight,
                                  float[] mergerNormBias, short[] mergerFc1Weight, float[] mergerFc1Bias,
                                  short[] mergerFc2Weight, float[] mergerFc2Bias) {
        this.config = Objects.requireNonNull(config, "config");
        this.patchWeight = Objects.requireNonNull(patchWeight, "patchWeight");
        this.patchBias = Objects.requireNonNull(patchBias, "patchBias");
        this.posEmbed = Objects.requireNonNull(posEmbed, "posEmbed");
        this.blocks = List.copyOf(Objects.requireNonNull(blocks, "blocks"));
        this.mergerNormWeight = Objects.requireNonNull(mergerNormWeight, "mergerNormWeight");
        this.mergerNormBias = Objects.requireNonNull(mergerNormBias, "mergerNormBias");
        this.mergerFc1Weight = Objects.requireNonNull(mergerFc1Weight, "mergerFc1Weight");
        this.mergerFc1Bias = Objects.requireNonNull(mergerFc1Bias, "mergerFc1Bias");
        this.mergerFc2Weight = Objects.requireNonNull(mergerFc2Weight, "mergerFc2Weight");
        this.mergerFc2Bias = Objects.requireNonNull(mergerFc2Bias, "mergerFc2Bias");
        validate();
    }

    public FlashNextVisionConfig config() { return config; }
    public short[] patchWeight() { return patchWeight; }
    public float[] patchBias() { return patchBias; }
    public float[] posEmbed() { return posEmbed; }
    public List<Block> blocks() { return blocks; }
    public float[] mergerNormWeight() { return mergerNormWeight; }
    public float[] mergerNormBias() { return mergerNormBias; }
    public short[] mergerFc1Weight() { return mergerFc1Weight; }
    public float[] mergerFc1Bias() { return mergerFc1Bias; }
    public short[] mergerFc2Weight() { return mergerFc2Weight; }
    public float[] mergerFc2Bias() { return mergerFc2Bias; }

    private void validate() {
        int hidden = config.hiddenSize(), inter = config.intermediateSize(), merged = config.mergedDim();
        requireLength("patch_embed.proj.weight", patchWeight.length, (long) hidden * config.patchDim());
        requireLength("patch_embed.proj.bias", patchBias.length, hidden);
        requireLength("pos_embed.weight", posEmbed.length, (long) config.numPositionEmbeddings() * hidden);
        if (blocks.size() != config.depth())
            throw new IllegalArgumentException("Expected " + config.depth() + " vision blocks, got " + blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            Block b = blocks.get(i);
            String p = "blocks." + i + ".";
            requireLength(p + "attn.qkv.weight", b.qkvWeight().length, 3L * hidden * hidden);
            requireLength(p + "attn.qkv.bias", b.qkvBias().length, 3L * hidden);
            requireLength(p + "attn.proj.weight", b.projWeight().length, (long) hidden * hidden);
            requireLength(p + "attn.proj.bias", b.projBias().length, hidden);
            requireLength(p + "mlp.linear_fc1.weight", b.fc1Weight().length, (long) inter * hidden);
            requireLength(p + "mlp.linear_fc1.bias", b.fc1Bias().length, inter);
            requireLength(p + "mlp.linear_fc2.weight", b.fc2Weight().length, (long) hidden * inter);
            requireLength(p + "mlp.linear_fc2.bias", b.fc2Bias().length, hidden);
            requireLength(p + "norm1.weight", b.norm1Weight().length, hidden);
            requireLength(p + "norm1.bias", b.norm1Bias().length, hidden);
            requireLength(p + "norm2.weight", b.norm2Weight().length, hidden);
            requireLength(p + "norm2.bias", b.norm2Bias().length, hidden);
        }
        requireLength("merger.norm.weight", mergerNormWeight.length, hidden);
        requireLength("merger.norm.bias", mergerNormBias.length, hidden);
        requireLength("merger.linear_fc1.weight", mergerFc1Weight.length, (long) merged * merged);
        requireLength("merger.linear_fc1.bias", mergerFc1Bias.length, merged);
        requireLength("merger.linear_fc2.weight", mergerFc2Weight.length, (long) config.outHiddenSize() * merged);
        requireLength("merger.linear_fc2.bias", mergerFc2Bias.length, config.outHiddenSize());
    }

    private static void requireLength(String name, long actual, long expected) {
        if (actual != expected)
            throw new IllegalArgumentException("Tensor " + name + " has " + actual + " elements, expected " + expected);
    }

    /** Expand BF16 bit patterns to floats (exact: BF16 values are representable in float32). */
    public static float[] toFloats(short[] bf16Bits) {
        float[] result = new float[bf16Bits.length];
        for (int i = 0; i < bf16Bits.length; i++) result[i] = FloatFormats.bf16(bf16Bits[i] & 0xFFFF);
        return result;
    }

    /** Read one BF16 tensor as raw bit patterns, validating dtype and shape. */
    public static short[] readBf16(SafeTensorIndex index, PagedTensorReader reader, String name,
                                   long... expectedShape) throws IOException {
        SafeTensorIndex.Tensor tensor = index.require(name);
        if (!tensor.dtype().equals("BF16"))
            throw new IOException("Expected BF16 tensor " + name + ", got " + tensor.dtype());
        List<Long> shape = tensor.shape();
        if (shape.size() != expectedShape.length)
            throw new IOException("Tensor " + name + " rank " + shape.size() + ", expected " + expectedShape.length);
        for (int i = 0; i < shape.size(); i++)
            if (shape.get(i) != expectedShape[i])
                throw new IOException("Tensor " + name + " shape " + shape + " disagrees with vision config");
        byte[] bytes = reader.read(tensor, 0, Math.toIntExact(tensor.bytes()));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[bytes.length / 2];
        for (int i = 0; i < result.length; i++) result[i] = buffer.getShort();
        return result;
    }

    private static float[] readVector(SafeTensorIndex index, PagedTensorReader reader, String name,
                                      long... expectedShape) throws IOException {
        return toFloats(readBf16(index, reader, name, expectedShape));
    }

    /**
     * Load all {@code model.visual.*} tensors. The caller owns the reader and may
     * close it after this returns; nothing is read lazily afterwards.
     */
    public static FlashNextVisionWeights load(FlashNextVisionConfig config, SafeTensorIndex index,
                                              PagedTensorReader reader) throws IOException {
        String root = "model.visual.";
        int hidden = config.hiddenSize(), inter = config.intermediateSize(), merged = config.mergedDim();
        short[] patchWeight = readBf16(index, reader, root + "patch_embed.proj.weight",
                hidden, config.inChannels(), config.temporalPatchSize(), config.patchSize(), config.patchSize());
        float[] patchBias = readVector(index, reader, root + "patch_embed.proj.bias", hidden);
        float[] posEmbed = readVector(index, reader, root + "pos_embed.weight", config.numPositionEmbeddings(), hidden);
        java.util.ArrayList<Block> blocks = new java.util.ArrayList<>(config.depth());
        for (int i = 0; i < config.depth(); i++) {
            String p = root + "blocks." + i + ".";
            blocks.add(new Block(
                    readBf16(index, reader, p + "attn.qkv.weight", 3L * hidden, hidden),
                    readVector(index, reader, p + "attn.qkv.bias", 3L * hidden),
                    readBf16(index, reader, p + "attn.proj.weight", hidden, hidden),
                    readVector(index, reader, p + "attn.proj.bias", hidden),
                    readBf16(index, reader, p + "mlp.linear_fc1.weight", inter, hidden),
                    readVector(index, reader, p + "mlp.linear_fc1.bias", inter),
                    readBf16(index, reader, p + "mlp.linear_fc2.weight", hidden, inter),
                    readVector(index, reader, p + "mlp.linear_fc2.bias", hidden),
                    readVector(index, reader, p + "norm1.weight", hidden),
                    readVector(index, reader, p + "norm1.bias", hidden),
                    readVector(index, reader, p + "norm2.weight", hidden),
                    readVector(index, reader, p + "norm2.bias", hidden)));
        }
        return new FlashNextVisionWeights(config, patchWeight, patchBias, posEmbed, blocks,
                readVector(index, reader, root + "merger.norm.weight", hidden),
                readVector(index, reader, root + "merger.norm.bias", hidden),
                readBf16(index, reader, root + "merger.linear_fc1.weight", merged, merged),
                readVector(index, reader, root + "merger.linear_fc1.bias", merged),
                readBf16(index, reader, root + "merger.linear_fc2.weight", config.outHiddenSize(), merged),
                readVector(index, reader, root + "merger.linear_fc2.bias", config.outHiddenSize()));
    }
}

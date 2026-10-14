package org.beehive.jitllm.model.vision;

import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * CPU reference forward pass of the Flash-Next (Qwen3-VL architecture) vision tower:
 * Conv3d patch embedding, bilinearly interpolated learned position embeddings
 * (linspace grid, merge-reordered), 27 pre-norm transformer blocks with full
 * (non-causal) attention per frame, 2D rotary position embedding (base 10000,
 * rotary dim = head_dim/2, angles repeated over both head halves, rotate-half),
 * and the spatial-merge projector (LayerNorm, merge grouping, GELU MLP) into the
 * language-model hidden size.
 *
 * <p>All arithmetic is float32; BF16 weights are expanded exactly. This is a
 * correctness reference, not a serving path: matrices are expanded per block and
 * attention materializes one score row at a time. There is no deepstack output
 * (the checkpoint's {@code deepstack_visual_indexes} is empty) and no window
 * attention (that is Qwen2.5-VL, not this tower).
 */
public final class FlashNextVisionTower {

    private static final float LAYER_NORM_EPS = 1e-6f;
    private static final double ROPE_THETA = 10000.0;

    private final FlashNextVisionConfig config;
    private final FlashNextVisionWeights weights;

    public FlashNextVisionTower(FlashNextVisionWeights weights) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.config = weights.config();
    }

    public FlashNextVisionConfig config() { return config; }

    /** Number of language-model tokens the given grid expands to after merging. */
    public int tokenCount(ImageGrid grid) { return config.tokenCount(grid); }

    /** Full tower: preprocessed image -&gt; {@code [tokenCount][outHiddenSize]} embeddings. */
    public float[][] forward(PreprocessedImage image) {
        Objects.requireNonNull(image, "image");
        return forward(image.pixelValues(), List.of(image.grid()));
    }

    /** Full tower over concatenated patches of one or more grids (packed, as in the reference). */
    public float[][] forward(float[] pixelValues, List<ImageGrid> grids) {
        float[][] hidden = patchEmbed(pixelValues, grids);
        float[][] pos = interpolatedPosEmbed(grids);
        for (int i = 0; i < hidden.length; i++)
            for (int j = 0; j < hidden[i].length; j++) hidden[i][j] += pos[i][j];
        double[][] cosSin = rotaryCosSin(grids); // [2][tokens * headDim]
        int[] cuSeqlens = cuSeqlens(grids);
        for (FlashNextVisionWeights.Block block : weights.blocks()) hidden = block(hidden, block, cosSin, cuSeqlens);
        return merge(hidden);
    }

    /** Patch embedding only (Conv3d with stride equal to kernel): {@code [patches][hidden]}. */
    public float[][] patchEmbed(PreprocessedImage image) {
        Objects.requireNonNull(image, "image");
        return patchEmbed(image.pixelValues(), List.of(image.grid()));
    }

    public float[][] patchEmbed(float[] pixelValues, List<ImageGrid> grids) {
        int patches = totalPatches(pixelValues, grids);
        float[][] input = unpack(pixelValues, patches, config.patchDim());
        return linear(input, FlashNextVisionWeights.toFloats(weights.patchWeight()), config.hiddenSize(), weights.patchBias());
    }

    // ── Transformer block ─────────────────────────────────────────────────────

    private float[][] block(float[][] x, FlashNextVisionWeights.Block block, double[][] cosSin, int[] cuSeqlens) {
        float[][] attended = attention(layerNorm(x, block.norm1Weight(), block.norm1Bias()), block, cosSin, cuSeqlens);
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[i].length; j++) x[i][j] += attended[i][j];
        float[][] mlpIn = layerNorm(x, block.norm2Weight(), block.norm2Bias());
        float[][] fc1 = linear(mlpIn, FlashNextVisionWeights.toFloats(block.fc1Weight()),
                config.intermediateSize(), block.fc1Bias());
        for (float[] row : fc1) for (int j = 0; j < row.length; j++) row[j] = geluTanh(row[j]);
        float[][] fc2 = linear(fc1, FlashNextVisionWeights.toFloats(block.fc2Weight()), config.hiddenSize(), block.fc2Bias());
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[i].length; j++) x[i][j] += fc2[i][j];
        return x;
    }

    private float[][] attention(float[][] x, FlashNextVisionWeights.Block block, double[][] cosSin, int[] cuSeqlens) {
        int tokens = x.length, hidden = config.hiddenSize(), heads = config.numHeads(), headDim = config.headDim();
        float[][] qkv = linear(x, FlashNextVisionWeights.toFloats(block.qkvWeight()), 3 * hidden, block.qkvBias());
        // qkv layout per token: [q (heads*headDim) | k | v]; apply 2D rotary to q and k.
        float[][] q = new float[tokens][hidden], k = new float[tokens][hidden], v = new float[tokens][hidden];
        for (int t = 0; t < tokens; t++) {
            System.arraycopy(qkv[t], 0, q[t], 0, hidden);
            System.arraycopy(qkv[t], hidden, k[t], 0, hidden);
            System.arraycopy(qkv[t], 2 * hidden, v[t], 0, hidden);
            applyRotary(q[t], cosSin, t);
            applyRotary(k[t], cosSin, t);
        }
        float[][] out = new float[tokens][hidden];
        double scale = 1.0 / Math.sqrt(headDim);
        for (int seg = 0; seg + 1 < cuSeqlens.length; seg++) {
            int start = cuSeqlens[seg], end = cuSeqlens[seg + 1], len = end - start;
            for (int head = 0; head < heads; head++) {
                int offset = head * headDim;
                for (int qi = 0; qi < len; qi++) {
                    float[] scores = new float[len];
                    float max = Float.NEGATIVE_INFINITY;
                    for (int ki = 0; ki < len; ki++) {
                        double dot = 0;
                        float[] qRow = q[start + qi], kRow = k[start + ki];
                        for (int d = 0; d < headDim; d++) dot += qRow[offset + d] * (double) kRow[offset + d];
                        scores[ki] = (float) (dot * scale);
                        if (scores[ki] > max) max = scores[ki];
                    }
                    double sum = 0;
                    for (int ki = 0; ki < len; ki++) { scores[ki] = (float) Math.exp(scores[ki] - max); sum += scores[ki]; }
                    float[] outRow = out[start + qi];
                    for (int ki = 0; ki < len; ki++) {
                        double weight = scores[ki] / sum;
                        float[] vRow = v[start + ki];
                        for (int d = 0; d < headDim; d++) outRow[offset + d] += (float) (weight * vRow[offset + d]);
                    }
                }
            }
        }
        return linear(out, FlashNextVisionWeights.toFloats(block.projWeight()), hidden, block.projBias());
    }

    private void applyRotary(float[] headPacked, double[][] cosSin, int token) {
        int headDim = config.headDim(), half = headDim / 2;
        double[] cos = cosSin[0], sin = cosSin[1];
        int base = token * headDim;
        float[] copy = headPacked.clone();
        for (int start = 0; start < headPacked.length; start += headDim)
            for (int d = 0; d < half; d++) {
                double c1 = cos[base + d], s1 = sin[base + d];
                double c2 = cos[base + half + d], s2 = sin[base + half + d];
                float left = copy[start + d], right = copy[start + half + d];
                headPacked[start + d] = (float) (left * c1 - right * s1);
                headPacked[start + half + d] = (float) (right * c2 + left * s2);
            }
    }

    // ── Merger ────────────────────────────────────────────────────────────────

    private float[][] merge(float[][] x) {
        int unit = config.spatialMergeSize() * config.spatialMergeSize();
        if (x.length % unit != 0)
            throw new IllegalArgumentException("Patch count " + x.length + " not divisible by merge unit " + unit);
        float[][] normed = layerNorm(x, weights.mergerNormWeight(), weights.mergerNormBias());
        int merged = config.mergedDim();
        float[][] grouped = new float[x.length / unit][merged];
        // Patches are already in merge-block-major order, so consecutive rows group directly.
        for (int i = 0; i < normed.length; i++)
            System.arraycopy(normed[i], 0, grouped[i / unit], (i % unit) * config.hiddenSize(), config.hiddenSize());
        float[][] fc1 = linear(grouped, FlashNextVisionWeights.toFloats(weights.mergerFc1Weight()), merged, weights.mergerFc1Bias());
        for (float[] row : fc1) for (int j = 0; j < row.length; j++) row[j] = geluErf(row[j]);
        return linear(fc1, FlashNextVisionWeights.toFloats(weights.mergerFc2Weight()), config.outHiddenSize(), weights.mergerFc2Bias());
    }

    // ── Position embeddings and rotary tables ─────────────────────────────────

    /** (row, col) of every patch in packed merge order, as the reference computes them. */
    static int[][] patchPositions(List<ImageGrid> grids, int mergeSize) {
        int total = 0;
        for (ImageGrid grid : grids) total += grid.patchCount();
        int[][] positions = new int[total][2];
        int index = 0;
        for (ImageGrid grid : grids) {
            if (grid.h() % mergeSize != 0 || grid.w() % mergeSize != 0)
                throw new IllegalArgumentException("Grid " + grid + " not divisible by merge size " + mergeSize);
            for (int frame = 0; frame < grid.t(); frame++)
                for (int blockRow = 0; blockRow < grid.h() / mergeSize; blockRow++)
                    for (int blockCol = 0; blockCol < grid.w() / mergeSize; blockCol++)
                        for (int inRow = 0; inRow < mergeSize; inRow++)
                            for (int inCol = 0; inCol < mergeSize; inCol++) {
                                positions[index][0] = blockRow * mergeSize + inRow;
                                positions[index][1] = blockCol * mergeSize + inCol;
                                index++;
                            }
        }
        return positions;
    }

    private float[][] interpolatedPosEmbed(List<ImageGrid> grids) {
        int side = config.numGridPerSide(), hidden = config.hiddenSize();
        float[] table = weights.posEmbed();
        int[][] positions = patchPositions(grids, config.spatialMergeSize());
        float[][] result = new float[positions.length][hidden];
        int index = 0;
        for (ImageGrid grid : grids) {
            double[] hCoords = linspace(side, grid.h()), wCoords = linspace(side, grid.w());
            for (int n = 0; n < grid.patchCount(); n++, index++) {
                int row = positions[index][0], col = positions[index][1];
                double hPos = hCoords[row], wPos = wCoords[col];
                int hFloor = (int) Math.floor(hPos), wFloor = (int) Math.floor(wPos);
                int hCeil = Math.min(hFloor + 1, side - 1), wCeil = Math.min(wFloor + 1, side - 1);
                double dh = hPos - hFloor, dw = wPos - wFloor;
                int[] corners = {hFloor * side + wFloor, hFloor * side + wCeil, hCeil * side + wFloor, hCeil * side + wCeil};
                double[] cornerWeights = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
                float[] out = result[index];
                for (int c = 0; c < 4; c++) {
                    int base = corners[c] * hidden;
                    double weight = cornerWeights[c];
                    for (int j = 0; j < hidden; j++) out[j] += (float) (weight * table[base + j]);
                }
            }
        }
        return result;
    }

    private static double[] linspace(int side, int count) {
        double[] coords = new double[count];
        if (count == 1) return coords;
        for (int i = 0; i < count; i++) coords[i] = (double) i * (side - 1) / (count - 1);
        return coords;
    }

    private double[][] rotaryCosSin(List<ImageGrid> grids) {
        int rotaryDim = config.rotaryDim(), headDim = config.headDim(), freqs = rotaryDim / 2;
        double[] invFreq = new double[freqs];
        for (int j = 0; j < freqs; j++) invFreq[j] = Math.pow(ROPE_THETA, -2.0 * j / rotaryDim);
        int[][] positions = patchPositions(grids, config.spatialMergeSize());
        double[] cos = new double[positions.length * headDim], sin = new double[positions.length * headDim];
        for (int t = 0; t < positions.length; t++) {
            int base = t * headDim;
            for (int half = 0; half < 2; half++) {
                int offset = base + half * rotaryDim;
                for (int j = 0; j < freqs; j++) {
                    double rowAngle = positions[t][0] * invFreq[j], colAngle = positions[t][1] * invFreq[j];
                    cos[offset + j] = Math.cos(rowAngle);
                    sin[offset + j] = Math.sin(rowAngle);
                    cos[offset + freqs + j] = Math.cos(colAngle);
                    sin[offset + freqs + j] = Math.sin(colAngle);
                }
            }
        }
        return new double[][]{cos, sin};
    }

    private static int[] cuSeqlens(List<ImageGrid> grids) {
        int segments = 0;
        for (ImageGrid grid : grids) segments += grid.t();
        int[] cu = new int[segments + 1];
        int index = 1;
        for (ImageGrid grid : grids)
            for (int frame = 0; frame < grid.t(); frame++) {
                cu[index] = cu[index - 1] + grid.h() * grid.w();
                index++;
            }
        return cu;
    }

    // ── Primitive ops ─────────────────────────────────────────────────────────

    private int totalPatches(float[] pixelValues, List<ImageGrid> grids) {
        Objects.requireNonNull(pixelValues, "pixelValues");
        Objects.requireNonNull(grids, "grids");
        if (grids.isEmpty()) throw new IllegalArgumentException("At least one grid is required");
        int total = 0;
        for (ImageGrid grid : grids) total += grid.patchCount();
        if (pixelValues.length != Math.multiplyExact(total, config.patchDim()))
            throw new IllegalArgumentException("pixelValues length " + pixelValues.length + " != patches "
                    + total + " * patchDim " + config.patchDim());
        // Every grid must also produce a whole number of merged tokens.
        for (ImageGrid grid : grids) config.tokenCount(grid);
        return total;
    }

    private static float[][] unpack(float[] flat, int rows, int width) {
        float[][] result = new float[rows][width];
        for (int i = 0; i < rows; i++) System.arraycopy(flat, i * width, result[i], 0, width);
        return result;
    }

    /** {@code out = x W^T + b} with {@code weight} row-major {@code [outFeatures][inFeatures]}. */
    static float[][] linear(float[][] x, float[] weight, int outFeatures, float[] bias) {
        int inFeatures = x[0].length;
        if (weight.length != Math.multiplyExact(outFeatures, inFeatures))
            throw new IllegalArgumentException("Weight length mismatch");
        float[][] result = new float[x.length][outFeatures];
        IntStream.range(0, x.length).parallel().forEach(row -> {
            float[] input = x[row], output = result[row];
            for (int o = 0; o < outFeatures; o++) {
                int base = o * inFeatures;
                double acc = bias[o];
                for (int i = 0; i < inFeatures; i++) acc += input[i] * (double) weight[base + i];
                output[o] = (float) acc;
            }
        });
        return result;
    }

    static float[][] layerNorm(float[][] x, float[] weight, float[] bias) {
        float[][] result = new float[x.length][];
        for (int r = 0; r < x.length; r++) {
            float[] row = x[r];
            double mean = 0;
            for (float value : row) mean += value;
            mean /= row.length;
            double variance = 0;
            for (float value : row) variance += (value - mean) * (value - mean);
            variance /= row.length;
            double invStd = 1.0 / Math.sqrt(variance + LAYER_NORM_EPS);
            float[] out = new float[row.length];
            for (int j = 0; j < row.length; j++) out[j] = (float) ((row[j] - mean) * invStd * weight[j] + bias[j]);
            result[r] = out;
        }
        return result;
    }

    /** {@code gelu_pytorch_tanh} activation used by the vision MLP blocks. */
    static float geluTanh(float x) {
        double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        return (float) (0.5 * x * (1.0 + Math.tanh(inner)));
    }

    /** Exact (erf) GELU used by the merger ({@code nn.GELU()} default). */
    static float geluErf(float x) {
        return (float) (0.5 * x * (1.0 + erf(x / Math.sqrt(2.0))));
    }

    private static double erf(double x) {
        // Abramowitz & Stegun 7.1.26, |epsilon| <= 1.5e-7.
        double sign = x < 0 ? -1 : 1;
        double ax = Math.abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        return sign * (1.0 - poly * Math.exp(-ax * ax));
    }
}

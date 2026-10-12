package org.beehive.jitllm.format.safetensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Row decoder for dense BF16, NVFP4 routed experts and block-FP8 MTP experts.
 * This is the storage reference; activation quantization belongs to the execution backend.
 */
public final class ModelOptMatrix {
    private final PagedTensorReader reader;
    private final SafeTensorIndex.Tensor weights, blockScales;
    private final int rows, columns, rowBytes, scaleColumns;
    private final float globalScale;
    private final MatrixExecutor executor;

    public ModelOptMatrix(SafeTensorIndex index, PagedTensorReader reader, String prefix) throws IOException {
        this(index,reader,prefix,null);
    }
    public ModelOptMatrix(SafeTensorIndex index, PagedTensorReader reader, String prefix, MatrixExecutor executor) throws IOException {
        this.reader = reader;
        this.executor = executor;
        weights = index.require(prefix + ".weight");
        if (weights.shape().size() != 2) throw new IOException("Expected weight matrix: " + prefix);
        rows = Math.toIntExact(weights.shape().get(0));
        int storedColumns = Math.toIntExact(weights.shape().get(1));
        switch (weights.dtype()) {
            case "U8" -> {
                columns = Math.multiplyExact(storedColumns, 2);
                if (columns % 16 != 0) throw new IOException("NVFP4 requires 16-wide blocks");
                blockScales = index.require(prefix + ".weight_scale");
                if (!blockScales.dtype().equals("F8_E4M3") || !blockScales.shape().equals(java.util.List.of((long) rows, (long) columns / 16)))
                    throw new IOException("Invalid NVFP4 block scales: " + prefix);
                globalScale = scalar(reader, index.require(prefix + ".weight_scale_2"));
                scaleColumns = columns / 16;
                rowBytes = storedColumns;
            }
            case "F8_E4M3" -> {
                columns = storedColumns;
                blockScales = index.require(prefix + ".weight_scale_inv");
                scaleColumns = (columns + 127) / 128;
                if (!blockScales.shape().equals(java.util.List.of((long) (rows + 127) / 128, (long) scaleColumns))
                        || !(blockScales.dtype().equals("BF16") || blockScales.dtype().equals("F32")))
                    throw new IOException("Invalid FP8 MTP block scales: " + prefix);
                globalScale = 1;
                rowBytes = columns;
            }
            case "BF16", "F16", "F32" -> {
                columns = storedColumns;
                blockScales = null;
                scaleColumns = 0;
                globalScale = 1;
                rowBytes = Math.multiplyExact(columns, weights.dtype().equals("F32") ? 4 : 2);
            }
            default -> throw new IOException("Unsupported ModelOpt matrix dtype: " + weights.dtype());
        }
        if (!Float.isFinite(globalScale) || globalScale <= 0) throw new IOException("Invalid global weight scale");
    }

    public int rows() { return rows; }
    public int columns() { return columns; }
    public String dtype() { return weights.dtype(); }
    public SafeTensorIndex.Tensor weights() { return weights; }
    public SafeTensorIndex.Tensor blockScales() { return blockScales; }
    public float globalScale() { return globalScale; }

    public void readRow(int row, float[] destination) throws IOException {
        if (row < 0 || row >= rows || destination.length != columns) throw new IllegalArgumentException("Invalid matrix row");
        byte[] packed = reader.read(weights, (long) row * rowBytes, rowBytes);
        if (weights.dtype().equals("U8")) {
            byte[] scales = reader.read(blockScales, (long) row * scaleColumns, scaleColumns);
            for (int column = 0; column < columns; column++) {
                int nibble = ((packed[column / 2] & 255) >>> ((column & 1) * 4)) & 15;
                destination[column] = FloatFormats.e2m1(nibble) * FloatFormats.e4m3(scales[column / 16]) * globalScale;
            }
        } else if (weights.dtype().equals("F8_E4M3")) {
            int elementBytes = blockScales.dtype().equals("BF16") ? 2 : 4;
            byte[] scales = reader.read(blockScales, (long) (row / 128) * scaleColumns * elementBytes, scaleColumns * elementBytes);
            ByteBuffer buffer = ByteBuffer.wrap(scales).order(ByteOrder.LITTLE_ENDIAN);
            for (int column = 0; column < columns; column++) {
                float scale = elementBytes == 2 ? FloatFormats.bf16(buffer.getShort((column / 128) * 2)) : buffer.getFloat((column / 128) * 4);
                destination[column] = FloatFormats.e4m3(packed[column]) * scale;
            }
        } else {
            ByteBuffer buffer = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
            for (int column = 0; column < columns; column++) destination[column] = switch (weights.dtype()) {
                case "BF16" -> FloatFormats.bf16(buffer.getShort());
                case "F16" -> Float.float16ToFloat(buffer.getShort());
                default -> buffer.getFloat();
            };
        }
    }

    public void multiply(float[] input, float[] output) throws IOException {
        if (input.length != columns || output.length != rows) throw new IllegalArgumentException("Matrix dimensions disagree");
        if(executor!=null) { executor.multiply(this,input,output); return; }
        multiplyReference(input,output);
    }
    /** BF16 projection boundary, with batch geometry supplied to hardware GEMM executors. */
    public void multiplyBf16(float[] input,float[] output,int prefillRows) throws IOException {
        if(input.length!=columns || output.length!=rows || prefillRows<=0) throw new IllegalArgumentException("Invalid BF16 projection geometry");
        if(executor!=null) executor.multiplyBf16(this,input,output,prefillRows);
        else {
            multiplyReference(input,output);
            for(int i=0;i<output.length;i++) output[i]=FloatFormats.roundBf16(output[i]);
        }
    }

    public void multiplyReference(float[] input,float[] output) throws IOException {
        if (input.length != columns || output.length != rows) throw new IllegalArgumentException("Matrix dimensions disagree");
        float[] row = new float[columns];
        for (int i = 0; i < rows; i++) {
            readRow(i, row);
            float sum = 0;
            for (int j = 0; j < columns; j++) sum = Math.fma(row[j], input[j], sum);
            output[i] = sum;
        }
    }

    /** Apply the global quantization alpha once after accumulation, as CUTLASS does. */
    public void multiplyNvfp4(Nvfp4Activation.Packed input,float[] output) throws IOException {
        if(!dtype().equals("U8") || input.values().length*2!=columns || input.scales().length!=columns/16 || output.length!=rows)
            throw new IllegalArgumentException("Invalid packed NVFP4 projection");
        if(executor!=null) { executor.multiplyNvfp4(this,input,output); return; }
        for(int row=0;row<rows;row++) {
            byte[] packed=reader.read(weights,(long)row*rowBytes,rowBytes);
            byte[] scales=reader.read(blockScales,(long)row*scaleColumns,scaleColumns);
            float sum=0;
            for(int column=0;column<columns;column++) {
                int shift=(column&1)*4;
                float w=FloatFormats.e2m1(((packed[column/2]&255)>>>shift)&15)*FloatFormats.e4m3(scales[column/16]);
                float x=FloatFormats.e2m1(((input.values()[column/2]&255)>>>shift)&15)*FloatFormats.e4m3(input.scales()[column/16]);
                sum=Math.fma(w,x,sum);
            }
            output[row]=sum*(globalScale*input.globalScale());
        }
    }

    private static float scalar(PagedTensorReader reader, SafeTensorIndex.Tensor tensor) throws IOException {
        if (!tensor.dtype().equals("F32") || tensor.elements() != 1) throw new IOException("Expected F32 scalar");
        return ByteBuffer.wrap(reader.read(tensor, 0, 4)).order(ByteOrder.LITTLE_ENDIAN).getFloat();
    }
}

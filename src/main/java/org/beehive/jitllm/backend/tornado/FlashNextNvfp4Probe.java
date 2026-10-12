package org.beehive.jitllm.backend.tornado;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.FlashNextWeightKernels;
import org.beehive.jitllm.format.safetensors.*;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Real packed NVFP4 or block-FP8 weight projection parity on a TornadoVM device. */
public final class FlashNextNvfp4Probe {
    public static Map<String, Object> run(Path model, String prefix, int deviceIndex) throws Exception {
        SafeTensorIndex index = SafeTensorIndex.open(model, name -> name.startsWith(prefix + "."));
        try (PagedTensorReader reader = new PagedTensorReader(1048576, 4096)) {
            ModelOptMatrix matrix = new ModelOptMatrix(index, reader, prefix);
            boolean nvfp4 = matrix.dtype().equals("U8");
            if (!nvfp4 && !matrix.dtype().equals("F8_E4M3")) throw new IllegalArgumentException("Probe requires NVFP4 or block-FP8 weights");
            var weight = index.require(prefix + ".weight");
            ByteArray weights = ByteArray.fromArray(reader.read(weight, 0, Math.toIntExact(weight.bytes())));
            FloatArray e2m1 = new FloatArray(16), e4m3 = new FloatArray(256);
            for (int i = 0; i < 16; i++) e2m1.set(i, FloatFormats.e2m1(i));
            for (int i = 0; i < 256; i++) e4m3.set(i, FloatFormats.e4m3(i));
            FloatArray input = new FloatArray(matrix.columns()), output = new FloatArray(matrix.rows());
            float[] x = new float[matrix.columns()], expected = new float[matrix.rows()];
            for (int i = 0; i < x.length; i++) { x[i] = (float) Math.sin(i * .017); input.set(i, x[i]); }
            matrix.multiply(x, expected);
            TaskGraph tasks = new TaskGraph("flashnextProjection");
            if (nvfp4) {
                var scale = index.require(prefix + ".weight_scale");
                ByteArray scales = ByteArray.fromArray(reader.read(scale, 0, Math.toIntExact(scale.bytes())));
                FloatArray globalScale = new FloatArray(1);
                globalScale.set(0, ByteBuffer.wrap(reader.read(index.require(prefix + ".weight_scale_2"), 0, 4)).order(ByteOrder.LITTLE_ENDIAN).getFloat());
                tasks.transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights, scales, e2m1, e4m3, globalScale)
                        .task("projection", FlashNextWeightKernels::nvfp4Matvec, input, weights, scales, e2m1, e4m3, globalScale, output, matrix.columns());
            } else {
                var scale = index.require(prefix + ".weight_scale_inv");
                ByteBuffer bytes = ByteBuffer.wrap(reader.read(scale, 0, Math.toIntExact(scale.bytes()))).order(ByteOrder.LITTLE_ENDIAN);
                FloatArray scales = new FloatArray(Math.toIntExact(scale.elements()));
                for (int i = 0; i < scales.getSize(); i++) scales.set(i, scale.dtype().equals("BF16") ? FloatFormats.bf16(bytes.getShort()) : bytes.getFloat());
                tasks.transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights, scales, e4m3)
                        .task("projection", FlashNextWeightKernels::blockFp8Matvec, input, weights, scales, e4m3, output, matrix.columns());
            }
            var graph = tasks.transferToHost(DataTransferMode.EVERY_EXECUTION, output).snapshot();
            var device = TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDevice(deviceIndex);
            double maxError = 0;
            long kernelNanos;
            try (var plan = new TornadoExecutionPlan(graph).withDevice(device).withProfiler(ProfilerMode.SILENT)) {
                kernelNanos = plan.execute().getProfilerResult().getDeviceKernelTime();
                if (kernelNanos <= 0) throw new AssertionError("No GPU kernel execution was recorded");
                for (int i = 0; i < expected.length; i++) {
                    double error = Math.abs(output.get(i) - expected[i]);
                    maxError = Math.max(maxError, error);
                    if (!Float.isFinite(output.get(i)) || error > 0.002 + 0.0002 * Math.abs(expected[i]))
                        throw new AssertionError("GPU projection mismatch at row " + i + ": " + output.get(i) + " vs " + expected[i]);
                }
            }
            return Map.of("matrix", prefix, "dtype", matrix.dtype(), "device", device.getPhysicalDevice().getDeviceName(), "rows", matrix.rows(),
                    "columns", matrix.columns(), "max_absolute_error", maxError, "packed_weight_bytes", weight.bytes(), "gpu_kernel_ns", kernelNanos);
        }
    }
}

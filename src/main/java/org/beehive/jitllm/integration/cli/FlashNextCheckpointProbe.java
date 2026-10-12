package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.format.safetensors.*;
import org.beehive.jitllm.model.loader.Qwen4PleTable;
import org.beehive.jitllm.server.Json;

/** Exercises real checkpoint loading, PLE hashing, and bounded NVMe row staging without a GPU. */
public final class FlashNextCheckpointProbe {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("Usage: FlashNextCheckpointProbe MODEL_DIR REQUEST.json");
        Map<String, Object> request = Json.parseObject(Files.readString(Path.of(args[1])));
        if(request.containsKey("activations")) {
            List<Object> outputs=new ArrayList<>();
            float scale=((Number)request.get("global_scale")).floatValue();
            for(Object row:(List<?>)request.get("activations")) {
                List<?> values=(List<?>)row;
                float[] input=new float[values.size()];
                for(int i=0; i<input.length; i++) input[i]=FloatFormats.roundBf16(((Number)values.get(i)).floatValue());
                var packed=Nvfp4Activation.quantize(input,scale);
                outputs.add(Map.of("values",bytes(packed.values()),"scales",bytes(packed.scales()),"dequantized",floats(packed.dequantize())));
            }
            System.out.println(Json.write(Map.of("outputs",outputs)));
            return;
        }
        String prefix = (String) request.getOrDefault("prefix", "model.language_model.layers.1.ple.ple_embedding");
        long budget = ((Number) request.getOrDefault("cache_bytes", 1048576)).longValue();
        int page = ((Number) request.getOrDefault("page_bytes", 4096)).intValue();
        int eos = ((Number) request.getOrDefault("eos", 248044)).intValue();
        String matrixPrefix = (String) request.get("matrix_prefix");
        SafeTensorIndex index = SafeTensorIndex.open(Path.of(args[0]), name -> name.startsWith(prefix + ".")
                || (matrixPrefix != null && name.startsWith(matrixPrefix + ".")));
        try (PagedTensorReader reader = new PagedTensorReader(budget, page)) {
            Qwen4PleTable table = new Qwen4PleTable(index, reader, prefix, eos);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("tensor_count", index.tensors().size());
            result.put("rows", table.rows());
            result.put("width", table.width());
            result.put("scale", (double) table.scale());
            List<Object> batches = new ArrayList<>();
            if (request.containsKey("windows")) {
                for (Object value : (List<?>) request.get("windows")) {
                    List<?> tokens = (List<?>) value;
                    int[] window = tokens.stream().mapToInt(v -> ((Number) v).intValue()).toArray();
                    long[] ids = table.hash().hash(window);
                    float[] gathered = new float[Math.multiplyExact(ids.length, table.width())];
                    table.gather(ids, gathered);
                    batches.add(Map.of("ids", Arrays.stream(ids).boxed().toList(), "values", floats(gathered)));
                }
            }
            result.put("batches", batches);
            if (matrixPrefix != null) {
                ModelOptMatrix matrix = new ModelOptMatrix(index, reader, matrixPrefix);
                List<Object> decoded = new ArrayList<>();
                for (Object row : (List<?>) request.get("matrix_rows")) {
                    float[] values = new float[matrix.columns()];
                    matrix.readRow(((Number) row).intValue(), values);
                    decoded.add(floats(values));
                }
                result.put("matrix", decoded);
            }
            if (request.containsKey("rows")) {
                long[] rows = ((List<?>) request.get("rows")).stream().mapToLong(v -> ((Number) v).longValue()).toArray();
                float[] gathered = new float[Math.multiplyExact(rows.length, table.width())];
                table.gather(rows, gathered);
                result.put("gather", floats(gathered));
            }
            var stats = reader.statistics();
            result.put("cache", Map.of("hits", stats.hits(), "misses", stats.misses(),
                    "bytes_read", stats.bytesRead(), "resident_bytes", stats.residentBytes(), "budget_bytes", stats.budgetBytes()));
            System.out.println(Json.write(result));
        }
    }

    private static List<Double> floats(float[] array) {
        List<Double> values = new ArrayList<>(array.length);
        for (float value : array) values.add((double) value);
        return values;
    }
    private static List<Integer> bytes(byte[] array) {
        List<Integer> output=new ArrayList<>(array.length);
        for(byte value:array) output.add(value&255);
        return output;
    }
}

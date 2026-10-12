package org.beehive.jitllm.format.safetensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.function.Predicate;

/** Metadata-only index of Hugging Face safetensors. No tensor is materialized at open. */
public final class SafeTensorIndex {
    public record Tensor(String name, String dtype, List<Long> shape, Path file,
                         long offset, long bytes) {
        public Tensor { shape = List.copyOf(shape); }
        public long elements() {
            long size = 1;
            for (long dim : shape) size = Math.multiplyExact(size, dim);
            return size;
        }
    }

    private final Map<String, Tensor> tensors;

    private SafeTensorIndex(Map<String, Tensor> tensors) {
        this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
    }

    public Collection<Tensor> tensors() { return tensors.values(); }
    public Tensor require(String name) {
        Tensor tensor = tensors.get(name);
        if (tensor == null) throw new IllegalArgumentException("Missing tensor: " + name);
        return tensor;
    }

    public static SafeTensorIndex open(Path directory) throws IOException {
        return open(directory, name -> true);
    }

    /** Select metadata for a subsystem without retaining every MoE expert descriptor. */
    public static SafeTensorIndex open(Path directory, Predicate<String> include) throws IOException {
        Path root = directory.toAbsolutePath().normalize();
        Path index = root.resolve("model.safetensors.index.json");
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        if (Files.exists(index)) {
            Map<Path, Set<String>> selected = indexedFiles(root, index, include);
            for (var file : selected.entrySet()) {
                Map<String, Tensor> header = readHeader(file.getKey());
                for (String name : file.getValue()) {
                    Tensor tensor = header.get(name);
                    if (tensor == null) throw new IOException("Index/header mismatch: " + name);
                    tensors.put(name, tensor);
                }
            }
        } else {
            readHeader(root.resolve("model.safetensors")).forEach((name, tensor) -> {
                if (include.test(name)) tensors.put(name, tensor);
            });
        }
        if (tensors.isEmpty()) throw new IOException("Empty safetensors model");
        return new SafeTensorIndex(tensors);
    }

    private static Map<Path, Set<String>> indexedFiles(Path root, Path index, Predicate<String> include) throws IOException {
        Map<String, Object> mapping = object(MetadataJson.parseObject(Files.readString(index)).get("weight_map"));
        Map<Path, Set<String>> selected = new LinkedHashMap<>();
        for (var entry : mapping.entrySet()) {
            if (!(entry.getValue() instanceof String file)) throw new IOException("Invalid shard filename");
            Path path = root.resolve(file).normalize();
            if (!path.startsWith(root)) throw new IOException("Shard escapes model directory: " + file);
            if (include.test(entry.getKey())) selected.computeIfAbsent(path, unused -> new LinkedHashSet<>()).add(entry.getKey());
        }
        return selected;
    }

    private static Map<String, Tensor> readHeader(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, length, 0);
            long headerSize = length.flip().getLong();
            if (headerSize < 2 || headerSize > 100_000_000 || headerSize > channel.size() - 8)
                throw new IOException("Invalid safetensors header size: " + file);
            ByteBuffer header = ByteBuffer.allocate(Math.toIntExact(headerSize));
            readFully(channel, header, 8);
            Map<String, Object> metadata = MetadataJson.parseObject(new String(header.array(), StandardCharsets.UTF_8));
            long base = 8 + headerSize;
            List<Tensor> ordered = new ArrayList<>();
            Map<String, Tensor> result = new LinkedHashMap<>();
            for (var entry : metadata.entrySet()) {
                if (entry.getKey().equals("__metadata__")) continue;
                Map<String, Object> value = object(entry.getValue());
                String dtype = (String) value.get("dtype");
                int width = switch (dtype) {
                    case "BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2" -> 1;
                    case "BF16", "F16", "I16", "U16" -> 2;
                    case "F32", "I32", "U32" -> 4;
                    case "F64", "I64", "U64" -> 8;
                    default -> throw new IOException("Unsupported dtype: " + dtype);
                };
                List<Long> shape = new ArrayList<>();
                for (Object dim : (List<?>) value.get("shape")) shape.add(nonnegativeLong(dim));
                List<?> offsets = (List<?>) value.get("data_offsets");
                if (offsets.size() != 2) throw new IOException("Invalid data_offsets");
                long begin = nonnegativeLong(offsets.get(0)), end = nonnegativeLong(offsets.get(1));
                if (begin > end || end > channel.size() - base) throw new IOException("Invalid tensor extent");
                Tensor tensor = new Tensor(entry.getKey(), dtype, shape, file, base + begin, end - begin);
                if (Math.multiplyExact(tensor.elements(), width) != tensor.bytes())
                    throw new IOException("Tensor byte count disagrees with shape: " + tensor.name());
                result.put(tensor.name(), tensor);
                ordered.add(tensor);
            }
            ordered.sort(Comparator.comparingLong(Tensor::offset));
            long end = base;
            for (Tensor tensor : ordered) {
                if (tensor.offset() != end) throw new IOException("Gap or overlapping tensor extent: " + tensor.name());
                end += tensor.bytes();
            }
            if (end != channel.size()) throw new IOException("Unindexed trailing safetensors data: " + file);
            return result;
        } catch (IllegalArgumentException | ArithmeticException | ClassCastException e) {
            throw new IOException("Invalid safetensors header: " + file, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> object(Object value) throws IOException {
        if (!(value instanceof Map<?, ?>)) throw new IOException("Expected a JSON object");
        return (Map<String, Object>) value;
    }

    private static long nonnegativeLong(Object value) throws IOException {
        if (!(value instanceof Number n)) throw new IOException("Expected integer");
        if (n instanceof Long l) {
            if (l < 0) throw new IOException("Negative extent: " + l);
            return l;
        }
        double d = n.doubleValue();
        // Floating JSON tokens may only denote exactly representable nonnegative extents.
        if (d < 0 || d > 9_007_199_254_740_991d || d != Math.rint(d)) throw new IOException("Invalid integer: " + d);
        return n.longValue();
    }

    public static void readFully(FileChannel channel, ByteBuffer destination, long offset) throws IOException {
        while (destination.hasRemaining()) {
            int read = channel.read(destination, offset);
            if (read < 0) throw new IOException("Unexpected EOF at " + offset);
            if (read == 0) throw new IOException("Read made no progress at " + offset);
            offset += read;
        }
    }
}

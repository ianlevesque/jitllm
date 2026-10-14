package org.beehive.jitllm.format.safetensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/** Bounded LRU of positional file reads; independent of the OS's file page cache. */
public final class PagedTensorReader implements AutoCloseable {
    private record Page(Path file, long index) {}
    public record Statistics(long hits, long misses, long bytesRead, long residentBytes, long budgetBytes) {}
    private final int pageBytes;
    private final int capacity;
    private final Map<Path, FileChannel> channels = new HashMap<>();
    private final LinkedHashMap<Page, byte[]> pages = new LinkedHashMap<>(16, 0.75f, true);
    private long hits, misses, bytesRead;
    private boolean closed;

    public PagedTensorReader(long budgetBytes, int pageBytes) {
        if (pageBytes < 1 || budgetBytes < pageBytes || budgetBytes / pageBytes > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Budget must hold at least one page");
        this.pageBytes = pageBytes;
        this.capacity = (int) (budgetBytes / pageBytes);
    }

    public int pageBytes() { return pageBytes; }
    public long budgetBytes() { return (long) capacity * pageBytes; }

    /** Return the resident page, filling it on a miss. Caller must hold the monitor. */
    private byte[] page(Page key) throws IOException {
        byte[] page = pages.get(key);
        if (page != null) { hits++; return page; }
        misses++;
        FileChannel channel = channels.get(key.file());
        if (channel == null) {
            channel = FileChannel.open(key.file(), StandardOpenOption.READ);
            channels.put(key.file(), channel);
        }
        long offset = key.index() * pageBytes;
        int count = (int) Math.min(pageBytes, channel.size() - offset);
        // Evict before allocating: the cache never temporarily holds capacity + 1 pages.
        if (pages.size() == capacity) pages.remove(pages.keySet().iterator().next());
        page = new byte[count];
        SafeTensorIndex.readFully(channel, ByteBuffer.wrap(page), offset);
        bytesRead += count;
        pages.put(key, page);
        return page;
    }

    public synchronized void read(SafeTensorIndex.Tensor tensor, long relativeOffset,
                                  byte[] destination, int start, int length) throws IOException {
        if (closed) throw new IOException("Reader is closed");
        Objects.checkFromIndexSize(start, length, destination.length);
        if (relativeOffset < 0 || length > tensor.bytes() || relativeOffset > tensor.bytes() - length)
            throw new IndexOutOfBoundsException("Read exceeds tensor: " + tensor.name());
        long absolute = tensor.offset() + relativeOffset;
        while (length > 0) {
            byte[] page = page(new Page(tensor.file(), absolute / pageBytes));
            int position = (int) (absolute % pageBytes);
            int count = Math.min(length, page.length - position);
            if (count <= 0) throw new IOException("Tensor page extends beyond file");
            System.arraycopy(page, position, destination, start, count);
            absolute += count;
            start += count;
            length -= count;
        }
    }

    /** Warm every page covering the range, in ascending offset order, without copying bytes out. */
    public synchronized void prefetch(SafeTensorIndex.Tensor tensor, long relativeOffset, long length) throws IOException {
        if (closed) throw new IOException("Reader is closed");
        if (relativeOffset < 0 || length < 0 || length > tensor.bytes() || relativeOffset > tensor.bytes() - length)
            throw new IndexOutOfBoundsException("Prefetch exceeds tensor: " + tensor.name());
        long absolute = tensor.offset() + relativeOffset;
        long remaining = length;
        while (remaining > 0) {
            byte[] page = page(new Page(tensor.file(), absolute / pageBytes));
            int position = (int) (absolute % pageBytes);
            long count = Math.min(remaining, page.length - position);
            if (count <= 0) throw new IOException("Tensor page extends beyond file");
            absolute += count;
            remaining -= count;
        }
    }

    public synchronized byte[] read(SafeTensorIndex.Tensor tensor, long offset, int length) throws IOException {
        byte[] bytes = new byte[length];
        read(tensor, offset, bytes, 0, length);
        return bytes;
    }

    public long[] readLongs(SafeTensorIndex.Tensor tensor) throws IOException {
        if (!tensor.dtype().equals("I64")) throw new IllegalArgumentException("Expected I64: " + tensor.name());
        ByteBuffer buffer = ByteBuffer.wrap(read(tensor, 0, Math.toIntExact(tensor.bytes()))).order(ByteOrder.LITTLE_ENDIAN);
        long[] result = new long[Math.toIntExact(tensor.elements())];
        for (int i = 0; i < result.length; i++) result[i] = buffer.getLong();
        return result;
    }

    public synchronized Statistics statistics() {
        long resident = 0;
        for (byte[] page : pages.values()) resident += page.length;
        return new Statistics(hits, misses, bytesRead, resident, (long) capacity * pageBytes);
    }

    @Override public synchronized void close() throws IOException {
        closed = true;
        pages.clear();
        IOException failure = null;
        for (FileChannel channel : channels.values()) {
            try { channel.close(); } catch (IOException e) { if (failure == null) failure = e; else failure.addSuppressed(e); }
        }
        channels.clear();
        if (failure != null) throw failure;
    }
}

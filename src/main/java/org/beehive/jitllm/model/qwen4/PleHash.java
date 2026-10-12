package org.beehive.jitllm.model.qwen4;

import java.util.Arrays;

/** Checkpoint-defined PLE XOR hashes, including EOS-delimited history. */
public final class PleHash {
    private final long[] multipliers, sizes, offsets;
    private final int eos, headsPerNgram;

    public PleHash(long[] multipliers, long[] sizes, long[] offsets, int eos) {
        if (multipliers.length < 2 || sizes.length == 0 || sizes.length != offsets.length
                || sizes.length % (multipliers.length - 1) != 0)
            throw new IllegalArgumentException("Invalid PLE hash layout");
        for (int i = 0; i < sizes.length; i++) {
            if (sizes[i] <= 0 || offsets[i] < 0) throw new IllegalArgumentException("Invalid PLE head extent");
            Math.addExact(offsets[i], sizes[i]);
        }
        this.multipliers = multipliers.clone();
        this.sizes = sizes.clone();
        this.offsets = offsets.clone();
        this.eos = eos;
        this.headsPerNgram = sizes.length / (multipliers.length - 1);
    }

    public int heads() { return sizes.length; }
    public int ngramSize() { return multipliers.length; }

    /** Window in chronological order, including the current token as its last element. */
    public long[] hash(int[] window) {
        if (window.length != multipliers.length) throw new IllegalArgumentException("Wrong PLE history width");
        long[] rows = new long[sizes.length];
        long mixed = (long) window[window.length - 1] * multipliers[0];
        boolean crossedEos = false;
        for (int position = 1; position < window.length; position++) {
            int previous = window[window.length - 1 - position];
            if (previous == eos) crossedEos = true;
            // Java overflow is the same two's-complement wrap as torch int64 multiply.
            mixed ^= (long) (crossedEos ? eos : previous) * multipliers[position];
            int start = (position - 1) * headsPerNgram;
            for (int h = start; h < start + headsPerNgram; h++) rows[h] = Math.floorMod(mixed, sizes[h]) + offsets[h];
        }
        return rows;
    }

    /** Per-sequence state. Snapshot/restore is required when speculative tokens are rejected. */
    public final class History {
        private final int[] window = new int[multipliers.length];
        public History() { reset(); }
        public void reset() { Arrays.fill(window, eos); }
        public long[] push(int token) {
            if (token < 0) throw new IllegalArgumentException("Negative token ID");
            System.arraycopy(window, 1, window, 0, window.length - 1);
            window[window.length - 1] = token;
            return hash(window);
        }
        public int[] snapshot() { return window.clone(); }
        public void restore(int[] state) {
            if (state.length != window.length) throw new IllegalArgumentException("Wrong history width");
            System.arraycopy(state, 0, window, 0, window.length);
        }
    }
}

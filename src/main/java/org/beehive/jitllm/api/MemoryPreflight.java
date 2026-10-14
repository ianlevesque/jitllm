package org.beehive.jitllm.api;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.memory.TornadoMemoryModel;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.model.qwen4.Qwen4Configuration;
import org.beehive.jitllm.runtime.memory.BufferClass;
import org.beehive.jitllm.runtime.memory.MemoryComponent;
import org.beehive.jitllm.runtime.memory.MemoryPlan;

/**
 * Builds a {@link MemoryPlan} from a model file, and refuses a load that cannot fit.
 *
 * <p>Package-private machinery behind {@link LocalModels#preflight}. It exists as its own type so
 * the facade stays a facade: this is where the backend is asked, and {@code LocalModels} does not
 * name {@code TornadoMemoryModel} or a device.
 *
 * <p>Flash-Next (Qwen4) safetensors directories take a dedicated path below. The generic path
 * cannot describe them: {@link ModelLoader#weightFootprint} reads GGUF metadata (it opens the
 * path as a file channel, which a checkpoint directory is not), and {@code TornadoMemoryModel}
 * sizes key/value storage as one private FP16/FP32 cache — for an FP8-paged sparse-attention
 * family with per-session state that both over-charges each element (4 B when FP8 was selected,
 * because the reservation knows only an fp16 flag) and under-charges the session count. The
 * Qwen4 figures are derived from the executor allocations themselves, documented per method.
 */
final class MemoryPreflight {

    private MemoryPreflight() {}

    /** The predicted plan for this file and these options. Reads descriptors, not tensor data. */
    static MemoryPlan plan(Path modelFile, int contextLength, ModelOptions options)
            throws IOException {
        Qwen4Configuration qwen4 = qwen4Configuration(modelFile, contextLength);
        if (qwen4 != null) {
            return qwen4Plan(qwen4, options);
        }
        // loadWeights = false, useTornadovm = false: the configuration comes from metadata and no
        // tensor is materialized, on the host or the device. That is what makes this a *pre*flight.
        var model = ModelLoader.loadModel(modelFile, contextLength, false, false);
        try (AutoCloseable resource = model instanceof AutoCloseable owned ? owned : () -> {}) {
        Configuration config = model.configuration();
        // Descriptors only. The loader owns the format types (Rule 4 permits it there and forbids
        // them here), and what comes back is a neutral footprint.
        //
        // The footprint has to know which representations this family keeps as they are, or it
        // predicts every quantized weight at its Q8_0 size — nearly double for a 4-bit file, which
        // is the difference between refusing a 27B model on a 24 GB device and running it.
        var weights =
                ModelLoader.weightFootprint(
                        modelFile,
                        org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry
                                .nativeDeviceTypes(model.architectureId()));
        return TornadoMemoryModel.predict(
                weights,
                config,
                options.executionPolicy(),
                TornadoDevices.current(),
                configuredBudgetBytes(),
                new org.beehive.jitllm.runtime.memory.KeyValueReservation(
                        options.maxConcurrentSessions(),
                        DelegatingModel.attachesSharedPool(
                                model,
                                options.storageOptions(),
                                org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection
                                        .mayHandle(
                                                model.architectureId(),
                                                loadedWeightType(modelFile),
                                                options.executionPolicy())),
                        org.beehive.jitllm.runtime.memory.KeyValueReservation.BLOCK_SIZE_TOKENS,
                        options.storageOptions().usesFp16KeyValueCache()));
        } catch (IOException | RuntimeException | Error failure) { throw failure; }
        catch (Exception failure) { throw new IOException("Cannot close preflight model", failure); }

    }

    /**
     * The weight type a load materializes, when the file's type determines it: F16 and Q8_0 are
     * loaded as they are. Anything else may be converted, so it is left unknown rather than
     * guessed.
     */
    private static org.beehive.jitllm.runtime.tensor.DataType loadedWeightType(Path modelFile) {
        try {
            return switch (org.beehive.jitllm.format.GgufModelFacts.read(modelFile).quant()) {
                case "F16" -> org.beehive.jitllm.runtime.tensor.DataType.F16;
                case "Q8_0" -> org.beehive.jitllm.runtime.tensor.DataType.Q8_0;
                default -> null;
            };
        } catch (IOException | RuntimeException unreadable) {
            return null;
        }
    }

    /**
     * Fails a known-over-capacity load before the first device allocation.
     *
     * <p>Silent when the budget is unknown or the plan's confidence is not exact. **Refusing on an
     * unreliable prediction would be worse than not predicting**: a conservative estimate that
     * happens to exceed the budget would block a load that would in fact have run, and the caller
     * has no way to overrule it. An under-confident plan is reported by {@link
     * LocalModels#preflight}, where a person can read it, rather than enforced here.
     *
     * <p>One Qwen4 exception is not a prediction at all: the Flash-Next device executors keep
     * hard byte budgets and check every session reservation against them at allocation time. When
     * the arithmetic below shows a reservation past such a budget, the load cannot succeed on any
     * device, so it is refused here with the plan attached instead of dying part-allocated.
     */
    static void refuseIfOverCapacity(Path modelFile, int contextLength, ModelOptions options) {
        MemoryPlan plan;
        try {
            plan = plan(modelFile, contextLength, options);
        } catch (IOException | RuntimeException e) {
            // A preflight that cannot be computed must not stop a load that might succeed. The
            // load's own error is the better diagnostic in that case.
            return;
        }
        try {
            Qwen4Configuration qwen4 = qwen4Configuration(modelFile, contextLength);
            String violation = qwen4 == null
                    ? null
                    : qwen4ExecutorCapViolation(qwen4, options.maxConcurrentSessions());
            if (violation != null) {
                throw new InsufficientDeviceMemoryException(
                        "Flash-Next cannot allocate this configuration: " + violation
                                + ".\n\n" + plan.describe(),
                        null);
            }
        } catch (IOException unreadable) {
            return;
        }
        if (plan.confidence() != MemoryPlan.Confidence.EXACT || plan.fitsConfiguredBudget()) {
            return;
        }
        throw new InsufficientDeviceMemoryException(plan);
    }

    // ── Flash-Next (Qwen4) plan ──────────────────────────────────────────────

    /**
     * The Qwen4 configuration of a safetensors directory, or {@code null} when this path is not
     * one. A directory that does claim to be Qwen4 but fails to parse (including a context request
     * past its native length) propagates that failure rather than falling through to the GGUF
     * path, whose error would name the wrong format.
     */
    static Qwen4Configuration qwen4Configuration(Path modelFile, int contextLength)
            throws IOException {
        Path configFile = modelFile.resolve("config.json");
        if (!Files.isDirectory(modelFile) || !Files.isRegularFile(configFile)) {
            return null;
        }
        try {
            return Qwen4Configuration.load(modelFile, contextLength);
        } catch (IOException | RuntimeException failure) {
            if (Files.readString(configFile).contains("\"qwen4_exp\"")) {
                throw failure;
            }
            return null;
        }
    }

    /**
     * A memory plan for the Flash-Next CUDA load, in the quantities that load actually allocates.
     *
     * <p>Checkpoint weights are not uploaded wholesale: the 51 GB PLE table and the expert tensors
     * are paged from NVMe and held in device caches bounded by {@code jitllm.qwen4.gpuCacheBytes}
     * (the {@code Qwen4GpuModel.load} budget split between projections and MoE experts), so the
     * weights line is that budget — a cap the caches evict under, not a resident measurement.
     * Everything per-session is multiplied by the admitted session count, because Flash-Next
     * sessions own their FP8 pages and recurrent state privately, and opening the Nth session
     * reserves the Nth copy whether or not the earlier ones have generated yet.
     *
     * <p>Confidence is CONSERVATIVE: the budgeted lines are caps and the per-session lines omit
     * native-array headers, so the total is a planning figure for a person to read, not a number
     * to refuse a load on. Refusal uses only the executor-budget arithmetic in {@link
     * #qwen4ExecutorCapViolation}, which mirrors checks the executors themselves make.
     */
    static MemoryPlan qwen4Plan(Qwen4Configuration config, ModelOptions options) {
        int sessions = options.maxConcurrentSessions();
        int checkpoints = Integer.getInteger("jitllm.qwen4.checkpointCapacity", 8);
        List<MemoryComponent> components = new ArrayList<>();
        components.add(new MemoryComponent(
                "weights (device caches, budgeted)",
                BufferClass.WEIGHTS_PER_LAYER,
                Long.getLong("jitllm.qwen4.gpuCacheBytes", 16L << 30), 1, 0));
        components.add(new MemoryComponent(
                "FP8 paged KV + compressed index (" + sessions + " session(s))",
                BufferClass.KV_CACHE,
                sessions * qwen4PagedKvBytesPerSession(config), 1, 0));
        components.add(new MemoryComponent(
                "QSA fixed buffers/workspaces (" + sessions + " session(s))",
                BufferClass.ACTIVATION_WORKSPACE,
                sessions * qwen4QsaFixedBytesPerSession(config), 1, 0));
        components.add(new MemoryComponent(
                "DeltaNet state + checkpoint banks (" + sessions + " session(s))",
                BufferClass.ACTIVATION_WORKSPACE,
                sessions * qwen4RecurrentBytesPerSession(config, checkpoints), 1, 0));
        components.add(new MemoryComponent(
                "PLE/HC layer state (budgeted)",
                BufferClass.ACTIVATION_WORKSPACE,
                Long.getLong("jitllm.qwen4.gpuLayerBytes", 64L << 20), 1, 0));
        if (Boolean.getBoolean("jitllm.qwen4.mtp") && config.mtpLayers() == 1) {
            var mtp = config.mtpConfiguration();
            components.add(new MemoryComponent(
                    "MTP FP8 paged KV + fixed buffers (" + sessions + " session(s))",
                    BufferClass.KV_CACHE,
                    sessions * (qwen4PagedKvBytesPerSession(mtp) + qwen4QsaFixedBytesPerSession(mtp)),
                    1, 0));
            components.add(new MemoryComponent(
                    "MTP executors (budgeted)",
                    BufferClass.WEIGHTS_PER_LAYER,
                    Long.getLong("jitllm.qwen4.gpuMtpLayerBytes", 64L << 20)
                            + Long.getLong("jitllm.qwen4.gpuMtpExpertBytes", 128L << 20)
                            + Long.getLong("jitllm.qwen4.gpuMtpFusionBytes", 128L << 20), 1, 0));
        }
        components.add(new MemoryComponent(
                "logits and sampling", BufferClass.RESULT,
                (long) config.vocabularySize() * Float.BYTES + Integer.BYTES, 1, 0));
        String violation = qwen4ExecutorCapViolation(config, sessions);
        return new MemoryPlan(
                components,
                configuredBudgetBytes(),
                MemoryPlan.Confidence.CONSERVATIVE,
                "flash-next CUDA; context " + config.contextLength()
                        + " of native " + config.contextLengthModel()
                        + "; FP8 E4M3 paged KV, session-private; " + sessions + " session(s)"
                        + "; checkpoint banks x" + checkpoints
                        + "; weights/PLE paged from checkpoint under device cache budgets"
                        + (violation == null ? "" : "; CANNOT FIT: " + violation));
    }

    /**
     * Tokens per QSA page: {@code QsaExecutor.Shape.pageTokens()} is {@code ratio * 512}, one
     * compressed-index block per {@code ratio} tokens and at most 512 blocks scored per page.
     */
    static int qwen4QsaPageTokens(Qwen4Configuration config) {
        return config.indexerCompressRatio() * 512;
    }

    /**
     * Bytes of paged key/value storage one session of one full-attention layer reserves at the
     * configured context, mirroring {@code FlashNextQsaExecutor.DeviceSession.Page}: per page, a
     * physical FP8 cache of {@code rows * (K+V) * kvDim} bytes, a BF16 compressed-index array of
     * {@code 512 * indexerHeadDim} elements, and — only for pages starting inside the fresh-prefill
     * window — a BF16 fresh shadow of the same shape as the cache (a one-element placeholder
     * elsewhere). The window is the executors' own constant: the first 4096 tokens of an initial
     * prefill attend through the shadow, everything past it through the FP8 pages.
     */
    static long qwen4PagedKvBytesPerLayerSession(Qwen4Configuration config) {
        int context = config.contextLength();
        int pageTokens = qwen4QsaPageTokens(config);
        long bytes = 0;
        for (int start = 0; start < context; start += pageTokens) {
            long rows = Math.min(pageTokens, context - start);
            long cacheRows = rows * 2L * config.kvDim();
            bytes = Math.addExact(bytes, cacheRows); // FP8: 1 byte per element
            bytes = Math.addExact(bytes, 512L * config.indexerHeadDim() * 2L);
            bytes = Math.addExact(bytes, start < 4096 ? cacheRows * 2L : 2L);
        }
        return bytes;
    }

    /** Paged KV bytes for one whole session: every full-attention layer owns its pages. */
    static long qwen4PagedKvBytesPerSession(Qwen4Configuration config) {
        return Math.multiplyExact(
                (long) config.keyValueLayerCount(), qwen4PagedKvBytesPerLayerSession(config));
    }

    /**
     * Fixed (context-independent) device buffers one QSA layer-session allocates beside its pages,
     * from the {@code FlashNextQsaExecutor.DeviceSession} constructor: projection/normalization
     * carriers, the gathered sparse KV window, score/probability arrays and their diagnostic trace
     * copies, the fused-score and XQA staging sets, and the two native library workspaces the
     * session reserves (1 MiB BF16/F32 GEMM, 16 MiB XQA). Carriers under a few KiB are folded into
     * the terms below rather than named; the workspaces dominate.
     */
    static long qwen4QsaFixedBytesPerLayerSession(Qwen4Configuration config) {
        long kvWidth = config.kvDim();
        long queryWidth = (long) config.numberOfHeads() * config.headSize();
        long indexQueryWidth = (long) config.indexerHeads() * config.indexerHeadDim();
        // Shape.selectedCapacity(), with the MTP draft tail as the upper bound for both configs.
        long selected = config.indexerBudget() + (long) config.indexerCompressRatio() - 1 + 3;
        long floats =
                2 * queryWidth + 2 * kvWidth // projected, key/value inputs
                        + queryWidth + kvWidth + queryWidth // query, key, gate
                        + indexQueryWidth + config.indexerHeadDim() // index input/query, pooled
                        + 2L * queryWidth // output, attended
                        + 2L * config.numberOfHeads() * selected // scores, probabilities
                        + 2L * config.numberOfHeads() * selected // their trace copies
                        + queryWidth + kvWidth + queryWidth + queryWidth // trace q/k/gate/attended
                        + 2L * config.headSize() + 2L * config.indexerHeadDim() + 256 + 512;
        long cap32 = (selected + 31) / 32 * 32;
        long cap64 = (selected + 63) / 64 * 64;
        long bfloats =
                selected * 2 * kvWidth // gathered sparse K+V
                        + config.numberOfKeyValueHeads() * cap32 * config.headSize() // packed keys
                        + config.numberOfKeyValueHeads() * 16L * config.headSize() // packed queries
                        + 2 * queryWidth // XQA query/output
                        + 2 * cap64 * kvWidth; // XQA keys/values
        long gemmScoreFloats = config.numberOfKeyValueHeads() * 16L * cap32;
        long ints = 3 * selected + 512 + cap64 / 64 + 12;
        return (floats + gemmScoreFloats) * Float.BYTES
                + bfloats * 2L
                + ints * Integer.BYTES
                + (1L << 20) // FlashNextBf16FloatLibrary.WORKSPACE_BYTES
                + (16L << 20); // FlashNextXqaLibrary.WORKSPACE_BYTES
    }

    /** Fixed QSA buffers for one whole session, across its full-attention layers. */
    static long qwen4QsaFixedBytesPerSession(Qwen4Configuration config) {
        return Math.multiplyExact(
                (long) config.keyValueLayerCount(), qwen4QsaFixedBytesPerLayerSession(config));
    }

    /**
     * Device bytes of DeltaNet state one session reserves, mirroring {@code
     * FlashNextDeltaNetExecutor.footprint}: each recurrent layer keeps its state plus {@code
     * checkpoints} snapshot banks of the same size (the rollback machinery speculative decoding
     * restores from), so the device figure is {@code checkpoints + 1} times the single state
     * {@link Qwen4Configuration#recurrentStateBytes()} reports for the delta part, plus the
     * executor's small carriers and the one-copy convolution/PLE remainder.
     */
    static long qwen4RecurrentBytesPerSession(Qwen4Configuration config, int checkpoints) {
        long recurrentLayers = config.numberOfLayers() - config.keyValueLayerCount();
        if (recurrentLayers <= 0) {
            return 0;
        }
        long channels = config.linearQkvDim();
        long stateFloats =
                (long) config.linearValueHeads() * config.linearKeyDim() * config.linearValueDim()
                        + channels * (config.linearConvKernel() - 1L);
        long floatsPerLayer =
                stateFloats * (checkpoints + 1L)
                        + channels * 2
                        + channels * config.linearConvKernel()
                        + (long) config.linearKeyHeads() * config.linearKeyDim() * 2
                        + (long) config.linearValueHeads() * config.linearValueDim() * 3
                        + config.linearValueHeads() * 4L
                        + config.linearValueDim();
        long deltaBytes = Math.multiplyExact(recurrentLayers, floatsPerLayer * Float.BYTES);
        long singleStateDelta =
                Math.multiplyExact(recurrentLayers, stateFloats * Float.BYTES);
        // PLE convolution state: the part of recurrentStateBytes() the delta layers do not own,
        // less their convolution windows (already inside stateFloats above, banked).
        long remainder = Math.max(0, config.recurrentStateBytes() - singleStateDelta);
        return Math.addExact(deltaBytes, remainder);
    }

    /**
     * Why this configuration cannot run under the Flash-Next executors' own byte budgets, or
     * {@code null} when every reservation fits. Each budget is read from the same system property
     * {@code Qwen4GpuModel.load} sizes the executor with, and each comparison is one the executor
     * itself makes when a session is created ({@code "QSA context reservation exceeds budget"},
     * {@code "GPU DeltaNet state exceeds budget"}) — failing here is certain, on any device, so
     * this is the one Qwen4 check {@link #refuseIfOverCapacity} enforces. The QSA reservation an
     * executor charges is its pages plus its fixed buffers, per layer-session.
     */
    static String qwen4ExecutorCapViolation(Qwen4Configuration config, int sessions) {
        long qsaBytes = sessions
                * (qwen4PagedKvBytesPerSession(config) + qwen4QsaFixedBytesPerSession(config));
        long qsaBudget = Long.getLong("jitllm.qwen4.gpuQsaBytes", 16L << 30);
        if (qsaBytes > qsaBudget) {
            return sessions + " session(s) at context " + config.contextLength()
                    + " reserve " + mib(qsaBytes) + " of FP8 QSA pages/buffers, over the "
                    + mib(qsaBudget) + " Flash-Next QSA budget"
                    + " (-Djitllm.qwen4.gpuQsaBytes); reduce the context length or the session count, or raise the budget";
        }
        int checkpoints = Integer.getInteger("jitllm.qwen4.checkpointCapacity", 8);
        long recurrentBytes = sessions * qwen4RecurrentBytesPerSession(config, checkpoints);
        long recurrentBudget = Long.getLong("jitllm.qwen4.gpuStateBytes", 4L << 30);
        if (recurrentBytes > recurrentBudget) {
            return sessions + " session(s) reserve " + mib(recurrentBytes)
                    + " of DeltaNet state and checkpoint banks, over the " + mib(recurrentBudget)
                    + " Flash-Next recurrent-state budget (-Djitllm.qwen4.gpuStateBytes);"
                    + " reduce the session count or raise the budget";
        }
        if (Boolean.getBoolean("jitllm.qwen4.mtp") && config.mtpLayers() == 1) {
            var mtp = config.mtpConfiguration();
            long mtpBytes = sessions
                    * (qwen4PagedKvBytesPerSession(mtp) + qwen4QsaFixedBytesPerSession(mtp));
            long mtpBudget = Long.getLong("jitllm.qwen4.gpuMtpQsaBytes", 2L << 30);
            if (mtpBytes > mtpBudget) {
                return sessions + " session(s) at context " + config.contextLength()
                        + " reserve " + mib(mtpBytes) + " of MTP FP8 QSA pages/buffers, over the "
                        + mib(mtpBudget) + " Flash-Next MTP QSA budget"
                        + " (-Djitllm.qwen4.gpuMtpQsaBytes); reduce the context length or the session count, or raise the budget";
            }
        }
        return null;
    }

    private static String mib(long bytes) {
        return String.format("%.1f MiB", bytes / 1048576.0);
    }

    /**
     * The backend's configured budget, or 0 when it is not set.
     *
     * <p>Read from the same property the backend charges against, so the preflight and the
     * allocator are talking about one number.
     */
    private static long configuredBudgetBytes() {
        String configured = System.getProperty("tornado.device.memory");
        if (configured == null || configured.isBlank()) {
            return 0;
        }
        try {
            String value = configured.trim().toUpperCase(java.util.Locale.ROOT);
            if (value.endsWith("B") && value.length() > 2) {
                int prefix = "KMGTPE".indexOf(value.charAt(value.length() - 2));
                if (prefix >= 0) {
                    long unit = (long) Math.pow(1024, prefix + 1);
                    return Long.parseLong(value.substring(0, value.length() - 2)) * unit;
                }
                return Long.parseLong(value.substring(0, value.length() - 1));
            }
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            // The backend rejects this value too, and its own message is the clearer one.
            return 0;
        }
    }
}

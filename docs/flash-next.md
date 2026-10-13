# JITLLM Flash-Next implementation

The goal is native JITLLM inference and serving of
`nvidia/Qwen3.8-Flash-Next-NVFP4`, with bounded NVMe PLE/Engram staging and the
features of the original launcher's `qwen-flash-next-staged` profile. The full 48-layer text pipeline now executes on the host and with native GPU
weight projections, DeltaNet state updates, HC/PLE arithmetic, QSA and MoE. The recorded 16-token ChatML
comparison now matches three greedy staged-reference tokens exactly; broader
full-model correctness remains to be qualified. The normal JITLLM loader/API can load the
checkpoint on CPU and CUDA through model-owned plans; production GPU serving is still under implementation.

This branch is based on upstream JITLLM
`67a7b8b28e19f3212bfbfc251de7ff9bb35bd863`. The implementation and service
registrations are committed directly under `src`; no launcher overlay is needed.
The tested CUDA/JDK 25 TornadoVM revision is
`a40ccf9a692fc06fe4c499f3700a3037e91dfebe`.

## Implemented and verified

- Safetensors index/header parsing, long file offsets, tensor extent validation,
  selective metadata loading, and positional file reads.
- A bounded LRU page reader. Its budget controls Java cache residency; the OS
  file page cache is separate. No copy of the PLE table is created.
- Numeric ordering of all 128 PLE shards, FP8/BF16 row decoding, BF16 scaling,
  checkpoint-defined signed-overflow XOR hashes, and EOS-delimited history.
  Per-sequence hash history supports reset, snapshot, and restore.
- NVFP4 low-nibble-first weight decoding with FP8 scales for each 16 elements
  and a second FP32 scale, plus the MTP's 128-by-128 FP8 block scaling.
- Packed NVFP4 and block-FP8 CUDA reference projections through TornadoVM.
  Both floating reference activations and packed W4A4 activations are supported.
  The scalar path remains available; target NVFP4 tensor-core execution is described below.
- A validated Qwen4 configuration with independent head/RoPE dimensions,
  explicit linear/full attention layers, one-based PLE placement, and separate
  MTP dimensions. Native context remains 262,144 rather than an inherited
  short-context default.
- A Hugging Face tokenizer adapter with NFC normalization, checkpoint regex,
  explicit BPE merge ranks, and added-token matching for ChatML, reasoning,
  vision and tools. IDs and decoding match HF on all 247 tested cases.
- BF16 host execution of hyper-connection normalization/mixing/injection and
  PLE key/value gating plus causal dilated convolution. PLE state supports
  exact snapshot/replay/reset, with bounded file staging throughout.
- Gated DeltaNet host execution with BF16 projections and causal convolution,
  L2-normalized Q/K, FP32 decay/delta recurrence, consecutive key/value head
  expansion, sigmoid-gated RMS normalization and output projection. Its convolution
  and temporal state support exact rollback/replay/reset.
- Optional TornadoVM CUDA DeltaNet execution: BF16 causal convolution,
  Q/K normalization, FP32 recurrence, and sigmoid-gated RMS normalization.
  Request-owned convolution/recurrent state stays on the GPU. A bounded device
  checkpoint pool handles nested rollback and reset without downloading recurrent
  state. The engine releases transaction checkpoints after every token, and API
  session close releases owned sequence resources. The host path remains available.
- Host QSA compression, sparse top-k block selection, GQA attention and MRoPE,
  with unfinished-block tails, fused normalization/RoPE precision and FP8-rounded
  KV values. Initial prefill uses a temporary BF16 view while retaining FP8-rounded
  persistent entries; the staged 4,096-token chunk boundary is honored. Prefill
  matches scaled-query BF16 rounding and tiled online softmax with BF16
  probabilities. Physical KV storage is currently FP32 on the host. Tests include a
  sparse selection beyond the 2,048-token budget and exact state rollback/reset.
- Host top-10/512-expert routing, NVFP4 W4A4 activation packing, per-layer global
  activation scales and the BF16 shared expert. Packed activation bytes match
  FlashInfer across four global scales. The host/scalar MoE gate remains red; the native cases below now pass exactly.
- A complete 48-layer host pipeline with transactional sequence rollback and
  a padded-vocabulary mask. The safetensors architecture and host forward pass
  are registered with JITLLM's existing provider services.
- A bounded GPU matrix cache, reusable projections, and private file mappings
  for packed weights. Graph reuse, eviction, plain/packed input switching, and
  unchanged source-file bytes are checked on the RTX 5090. The cache budget
  accounts for logical buffers; driver allocations and the OS page cache are
  separate. PLE remains with the 1 MiB positional-read cache.
- Normal `LocalModels.load` API loading, three independent sessions, admission
  refusal for a fourth session, reset, refusal to close with live sessions, and
  checkpoint file-handle release at model close. The legacy single-type weight
  API reports BF16 dense projections; per-matrix NVFP4 descriptors retain the
  expert storage format.

`benchmarks/jitllm-flash-next/checkpoint-parity.json` records exact PyTorch parity
for 67 hash windows, 1,584 gathered rows including all shard boundaries, and
30 decoded rows across six target/MTP expert matrices. The probe ran with a
128 MiB Java heap and a 1 MiB row cache; it did not load the 51.2 GB table.
`gpu-projection-parity.json` records all six packed projections on the RTX 5090,
including positive CUDA kernel timings and zero difference from the Java
reference. These checks do not prove full-model output correctness.
`metadata-parity.json` records configuration/tokenizer checks.
`ple-hyper-parity.json` checks actual BF16 weights: hyper-connection outputs
match exactly, and the 12-token PLE sequence has small BF16 rounding
differences (maximum observed absolute difference 0.00006103515625). It includes EOS
boundaries, the full dilated convolution history, and exact state rollback and
reset. Host model integration is complete; full device execution remains.
`deltanet-parity.json` checks 12 sequential steps with real layer-0 weights
against an independent PyTorch recurrence. Maximum observed absolute output
difference is 0.0009765625; convolution and recurrent rollback/reset checks are
exact. This report covers the host path.
`gpu-deltanet-parity.json` and `gpu-deltanet-projected-parity.json` now cover the
CUDA recurrence against the same independent PyTorch reference, with host and
GPU projections respectively. Both pass 12 steps at the original rtol 0.008 /
atol 0.001; maximum observed error is 0.0009765625. The CUDA probe checks nested
checkpoint replay, exact reset, independent sequences, pool exhaustion, rejected
foreign/released checkpoints, and zero tracked memory after close. The projected
check also verifies state-budget refusal. Neither path downloads recurrent state.

`qsa-fp8-fused-parity.json` covers 12 steps with mixed MRoPE coordinates,
FP8-rounded KV, exact sparse selections and state replay/reset. Its gate oracle
uses SGLang’s actual CUDA fused sigmoid/multiply kernel.
`qsa-fresh-prefill-parity.json` checks actual SGLang CUDA sparse prefill, including
rollback across release of the fresh BF16 view.
`activation-parity.json` records exact NVFP4 activation bytes against FlashInfer.
`real-activation-parity.json` also checks the real first-layer mixed input at its
checkpoint calibration scale; packed values and block scales match exactly.
`gpu-cache-parity.json` records 21 GPU projections, nine uploads and eight
cache evictions within a 7,500,000-byte budget; packed W4A4, block-FP8 and BF16
results match the storage reference (floating NVFP4 inputs differ by at most
0.0000016689300537109375).
`api-lifecycle.json` records three sessions at context 262,144 and zero leaked
checkpoint descriptors after close.

`moe-parity.json` intentionally records **passed=false**: two of 2,560 values
in one step exceed rtol 0.008 / atol 0.0001. The oracle uses actual FlashInfer
CUTLASS and SGLang routing, with fused finalization disabled as in the staged
profile. Its maximum observed absolute difference is 0.0001220703125.
`full-gpu-projected-conformance.json` records **passed=true** for one short
text case. Native and uncached staged SGLang process the same 16-token ChatML
prompt and emit exactly `[17, 478, 220]` (`2 + `) for `2+2=?`. Native runs 39,819
GPU projections with the PLE reader bounded to 1 MiB. This is a GPU-projected
host pipeline; it does not prove full GPU serving or the remaining profile
requirements. `full-conformance-pre-sigmoid.json` preserves the earlier failure.
The decisive correction was using the checkpoint's sigmoid recurrent output
gate rather than the inherited SiLU default. Fresh BF16 prefill K/V and fused
attention arithmetic were also corrected.

`full-gpu-recurrence-conformance.json` repeats that same short case with CUDA
DeltaNet enabled. It matches a fresh uncached staged result exactly, executes
648 recurrent steps and 648 device checkpoints across the 36 recurrent layers,
and leaves no live transaction checkpoints. The recurrent executor tracks
1,071,175,968 bytes including eight checkpoint slots per layer and releases all
of it on sequence close. No recurrent state is downloaded. This case took about
192 seconds on the RTX 5090. HC, PLE, QSA and MoE nonlinear operations remain
on the host; broader qualification and full GPU serving are still pending.
`full-gpu-recurrence-rollback-conformance.json` checks that same prompt after
injecting an error after layer 20 of its first token. All 36 device recurrent
states restore, the sequence returns to position zero, transaction checkpoints
are released, and retry still matches the uncached reference exactly. This is a
failure/retry check for the same short case, not broader text qualification.



CUDA HC normalization, low-rank activation, gated mixing and residual injection,
and PLE normalization, signed-root gating, dilated convolution and residual add
are now optional native operations. PLE history remains request owned on the
GPU, with bounded device checkpoints and exact rollback/reset. Only gathered
Engram rows and projection outputs are transferred; the 51.2 GB table remains
in the positional-file reader. These operations still exchange activation
vectors with the host between projection plans; device-resident plan fusion is
remaining work.

`gpu-hc-ple-parity.json` checks real layer-1 weights against the staged HC CUDA
implementation and an independent BF16 PLE sequence. HC is exact and the maximum
PLE difference is 0.00006103515625 at the unchanged rtol 0.008 / atol 0.0001.
`gpu-hc-row-boundaries.json` covers 1, 16, 17, 32, 33, 64, 1,024 and 4,096-row
geometries, with SGLang's actual fused PLE decode gate. Every HC comparison is
exact. It also checks graph replay, independent PLE states, checkpoint pool and
buffer budgets, stale/foreign checkpoint rejection, residual add, closed-state
refusal and zero tracked buffers after close. These are subsystem checks, not
full-model long-context qualification.

Native HC residual injection now reads the small BF16 injection matrix once
into a budgeted, model-owned GPU buffer and computes its gate projection inside
the CUDA combine operation. Its reduction matches the staged kernel: eight
values per lane, paired products, descending warp shuffles and eight ordered
warp sums, followed by FP32 sigmoid and a fused residual add. Unsupported shapes
keep the projection interface. The new backend-neutral overload has a default
implementation for existing providers.

`gpu-hc-native-combine-rows-{1,17,33,1024}.json` checks both sides of the staged
split-combine boundary and large prefill. HC normalization, mix and combine
outputs are exact in all four cases. PLE's original tolerance, rollback/reset,
independent sequence checks and resource budgets still pass. Synthetic teacher inputs using real MTP weights also pass the original
activation tolerance; remaining failures start in draft attention.

SM120 HC mixing uses F32 projection accumulators for <=16 rows. Larger prefill
chunks use native BF16 GEMM with the actual batch geometry and a 32 MiB cuBLAS
workspace, matching the staged runtime. HC combine retains F32 accumulation
across the tested row counts. The sequential diagnostic runner repeats its
current row to obtain the same GEMM geometry; efficient batched prefill still
needs to use distinct rows and keep activations on the device. The separate
BF16 GEMM caches together are capped at min(main matrix cache budget, 256 MiB),
including workspaces: three quarters for BF16 stores and one quarter for FP32
stores. Plans are owned by their caller thread so stream changes cannot reset
a shared cuBLAS workspace.

`gpu-bf16-batch-parity.json` verifies the distinct-row native GEMM primitive on
both real HC matrices at batches 1, 33 and 64. All six cases match PyTorch
exactly, including graph reuse, caller-thread changes, checkpoint-byte
preservation and zero tracked memory after close. This adds BF16 tensor-core
execution; packed NVFP4 tensor-core execution is still required.
The `pre-gemm` and `pre-workspace` HC reports preserve the earlier strict
failures at 33 and 1,024 rows and the traces that identified their causes.

Small-row HC down/up projections now use BF16 tensor-core operands with FP32
output. Down projection splits K into 256-element partitions and adds the FP32
partials in order; up projection keeps the full K dimension. This avoids an
intermediate BF16 store. `gpu-hc-f32-projection-portable-parity.json` checks the
real target and MTP attention-HC matrices across three inputs each: all 12
results match CUDA BF16/FP32 GEMMs exactly, including replay, caller-thread
changes and resource release. Run it with
`verify_jitllm_hc_projection.py` through the portable runtime wrapper.

Fused QSA score calculation now packs selected BF16 keys and grouped queries
for native BF16/FP32 GEMMs. Prefill retains the staged BF16-scaled query;
decode scales the FP32 result. Shapes with unsupported tensor-core dimensions
retain the scalar score calculation. Optional stage tracing downloads copied
normalization, score, probability and attention buffers only when requested.

`gpu-qsa-tensorcore-fresh-regression.json` and
`gpu-qsa-tensorcore-decode-regression.json` repeat both existing 12-step real-layer
checks at the original rtol 0.016 / atol 0.004. Both pass with a maximum absolute
output difference of 0.0078125, exact rollback/reset and zero reservations after
close. `gpu-qsa-draft-tensorcore-regression.json` repeats the frozen-selection
page boundary check and passes. `gpu-bf16-cache-split-regression.json` repeats
all six distinct-row BF16 GEMMs after splitting the cache; outputs remain exact.

`full-gpu-hc-ple-rollback-conformance.json` repeats the 16-token ChatML case
with GPU HC/PLE and DeltaNet enabled, including an injected layer-20 failure.
It restores all 36 recurrent states and the PLE state, releases transaction
checkpoints, and still emits `[17, 478, 220]` exactly like a fresh uncached
staged request. `full-gpu-hc-ple-bf-prefill-conformance.json` adds a 17-token
prefill that executes 3,298 native BF16 GEMMs; its three greedy tokens also
match the uncached reference. Both keep the file reader at 1 MiB and release
tracked HC/PLE buffers at engine close. These two short cases do not establish
full GPU serving, broad text accuracy or full-context/concurrent inference.
The sequential runs took about 213 and 259 seconds on the RTX 5090.

Optional CUDA QSA now stores persistent K/V in physical FP8 bytes (or BF16
bytes), with BF16 compressed keys, GPU block scoring/top-k selection and
sparse GQA. The executor reserves a bounded full-context address space and
materializes 2,048-token device pages when reached. Snapshot metadata keeps a
prefix length and the small unfinished compression input on the host; KV is
append-only and never downloaded. Diagnostic selections are downloaded. The
first 4,096-token BF16 shadow is logically released after fresh prefill and
physically retained until session close so rollback can recover it.

`gpu-qsa-fp8-parity.json` and `gpu-qsa-fresh-prefill-parity.json` check 12 steps
using real layer-3 weights, mixed MRoPE coordinates and GPU projections.
Both pass the existing rtol 0.016 / atol 0.004, with a maximum observed
absolute difference of 0.0078125, matching sparse selection sets, exact
rollback/reset, positive CUDA kernel timings and zero tracked reservation
after close. The fresh reference uses SGLang's CUDA sparse GQA kernel.
These layer checks cover a single page. `gpu-qsa-paging-parity.json` uses
reduced head widths to execute 4,099 tokens through three 2,048-token pages
within a reserved 262,144-token context. Its eight boundary cases match the
host oracle exactly, including merging more than 512 compressed blocks,
unfinished tails and the fresh-to-FP8 transition at token 4,096. It checks
snapshot ownership, disposed/stale/future checkpoints, checkpoint and buffer
budgets, independent sequences, context overflow, exact reset and closed-state
refusal. Only three pages materialize; all tracked reservations are released.
This subsystem check does not prove full-model long-context inference.

`full-gpu-qsa-rollback-conformance.json` enables CUDA QSA in the full 48-layer
runner alongside GPU projections, DeltaNet and HC/PLE. After an injected
layer-20 failure, all 12 QSA states restore and the same 16-token ChatML case
emits `[17, 478, 220]` exactly like a fresh uncached staged request. It executes
221 QSA forwards with no KV downloads or live checkpoints, and sequence close
releases the 3,580,597,824-byte full-context reservation. Reached QSA pages
account for 77,071,536 bytes, separate from shared sparse workspace buffers.
The positional-file reader remains bounded to 1 MiB. This short case does not
qualify long context, concurrent inference, speculative decoding or serving.
The normal CUDA model provider now uses these executors through model-owned plans; layer-to-layer plan fusion remains required.

`full-gpu-qsa-bf-prefill-conformance.json` also matches the fresh staged tokens
for the 17-token prefill case, with 228 QSA forwards and 3,298 native BF16 HC
GEMMs. It releases QSA, recurrent and PLE state at sequence/engine close.
The two cases take about 229 and 261 seconds in the sequential diagnostic
runner. They establish these short cases only; efficient batched prefill,
plan fusion, broader text correctness and production GPU serving remain.



Native NVFP4 projections now use CUDA 13 cuBLASLt with packed E2M1 weights,
E4M3 scales for each 16 elements, and BF16 outputs. CUDA also packs activations
using the staged fast-reciprocal encode recipe; no BF16 weight expansion occurs.
Scale bytes are rearranged into the vendor's 128x4 layout on the GPU. The binding
uses TornadoVM's library-provider SPI, device alpha/beta pointers, cached native
descriptors and a 1 MiB workspace per cached plan. GEMM batches contain distinct
rows and zero padding to a multiple of 32. The encoding plan uses
`--prec-div=false`; other native plans retain their existing precision policy.
Runtime encoder loop bounds avoid repeated Graal expansion of nested loops.

`gpu-nvfp4-native-batch-parity.json` covers the real layer-0 expert gate/up/down
matrices at batches 1, 3, 33 and 128. All 12 cases match FlashInfer CUTLASS
exactly, including packed activation/scaling bytes, graph replay, caller-thread
changes, raw/packed input switching, device alpha changes, source-byte windows
and zero native contexts/descriptors after close. The comparison retains
rtol 0.008 / atol 0.0001. Explicit packing diagnostics download bytes; normal
projections retain their encoded activations on the device.

`gpu-native-nvfp4-moe-parity.json` passes all three strict real MoE cases exactly,
with 90 native GEMMs and 105 total projections. This resolves the recorded
numeric difference for these native cases. `moe-parity.json` remains the
historical failing host/scalar comparison. This standalone projection path retains host routing, gates and finalization.
The complete device MoE implementation below removes those boundaries; broader
trajectories remain unqualified. `gpu-nvfp4-native-pre-quantization-fix.json` preserves the three
activation-byte mismatches that identified the encode rounding problem.

Set `gpu_nvfp4: true` with `gpu_projections: true` to select this native path.
The main `gpu_cache_bytes` budget is split equally between packed native plans
(including their workspaces) and dense/scalar projections; the separate BF16
GEMM cache remains capped at 256 MiB. The normal CUDA model provider is integrated; cross-layer device
plan fusion, efficient batched full-model prefill, complete MTP draft layers,
long-context/concurrent serving and multimodal/chat replay remain required.
`full-gpu-nvfp4-rollback-conformance.json` checks the same 16-token ChatML
failure/retry case with native NVFP4 enabled. Its three greedy tokens match a
fresh uncached staged result exactly. It executes 26,550 native GEMMs, restores
all recurrent/QSA/PLE states after the injected error, and releases native
contexts/descriptors and sequence buffers after close. The file reader remains
at 1 MiB, and both projection caches together stay within 16 GiB. This is a
short sequential diagnostic, not serving or broad text qualification. It took
about 403 seconds; the pre-loop-bound report retains the earlier 461-second
run. Layer-plan fusion and grouped expert execution remain needed for practical
latency and memory use.

The binding follows [NVIDIA's CUDA 13 cuBLASLt block-scaling requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#d-block-scaling-factors-layout).


MoE routing, target/shared SwiGLU gates and finalization now execute on CUDA.
Each layer retains shared/router BF16 weights and reusable compute plans. After
routing, only selected expert IDs/weights cross back to stage cache entries;
expert intermediates remain on the device. The expert LRU keeps packed up/gate
and down matrices independently of the compute plans, with pinned leases
through forward execution. Up/gate matrices share one native GEMM. Selected
GEMMs enqueue on the same stream using cached descriptors and one 1 MiB native
workspace per layer. CUDA 13 pointer-array batching does not support these
block scales, so this uses repeated native launches inside one library task.

`gpu-device-moe-parity.json` checks the same three strict MoE cases with full
device math and a 52 MiB total budget. All outputs are exact; replay is exact,
five cache evictions preserve results, intermediate downloads remain zero,
and resources return to zero after close. Only one layer plan is built across
six forwards; the 120 native GEMMs do not construct per-expert compute plans.
The existing native-GEMM compatibility suite still passes all 12 cases exactly.

`full-gpu-device-moe-rollback-conformance.json` and
`full-gpu-device-moe-bf-prefill-conformance.json` match fresh uncached staged
output `[17, 478, 220]` for the 16-token failure/retry and 17-token BF16-prefill
cases. They run 885/912 device MoE forwards and 17,700/18,240 native expert
GEMMs, with no expert-intermediate downloads. Engine/factory close releases
MoE layer buffers, expert registrations and native workspaces/descriptors.
The file reader remains at 1 MiB. The recorded runs take about 104 and 150
seconds; the earlier native diagnostic took about 403 seconds. These are short
case timings, not a general performance benchmark or production qualification.

Set `gpu_moe: true` with `gpu_projections: true` in the diagnostic request.
The main cache budget is split equally between dense/scalar projections and
MoE resources, including shared/router weights, intermediate buffers, native
workspace and packed expert residency. The standalone native NVFP4 projection
path remains available. Full layer-to-layer device plan fusion, efficient
batched prefill, MTP/FP8 draft math, native vision
and exact chat replay, full context/concurrency and deployment are still needed.

## Native block-FP8 MTP expert primitives

MTP routed experts now have native W8A8 CUDA execution that preserves the
checkpoint's E4M3 bytes and 128-by-128 FP32 scales. On SM120, CUDA 13's direct
FP32 block-scale descriptors are unavailable, so cuBLASLt computes FP32
partials for each 128-wide K partition. CUDA applies the original weight and
activation scales, accumulates the partitions, and stores BF16 results. Weights
are never expanded to BF16. The distinct-row projection primitive rounds inputs
to BF16 before dynamic group encoding.

The MTP MoE executor keeps routing, activation encoding, SwiGLU, FC1/FC2
partials, the shared expert and finalization on the device. Only routing
metadata and final output are downloaded during normal forwards. Packed expert
residency uses bounded LRU entries with leases through each forward. FC2 applies
the routing weight before its BF16 store, matching the staged Triton path.
Optional stage traces explicitly download intermediates for diagnosis.

The CUDA encoder uses `--use_fast_math`, as does the staged SM120 quantizer.
Precise division changed E4M3 rounding at midpoint boundaries and failed strict
parity; using the staged fast reciprocal fixes the observed mismatches. The
host reference retains IEEE division and can differ at those boundaries.

`gpu-fp8-native-batch-parity.json` records 12 real MTP gate/up/down projections
at batches 1, 3, 33 and 128. Activation bytes and scales match exactly in every
case. Eleven output cases are exact; the remaining case differs in one value by
0.0000152587890625 and passes the unchanged rtol 0.008 / atol 0.0001. Graph replay
and a changed caller thread are exact, and native resources return to zero.
`gpu-fp8-moe-parity.json` records three complete real MTP MoE cases, each
replayed exactly. Two outputs are exact; the third has maximum error
0.00006103515625 and passes the same strict tolerance. Six forwards reuse one
layer plan, execute 1,500 native partition GEMMs and force five expert-cache
evictions. Peak tracked residency is 132,353,572 bytes within a 128 MiB budget;
normal intermediate downloads and all resources after close are zero.
`fp8-regression-tests.json` records 808 Java tests with zero failures/errors
and three skips, 112 passing launcher tests, and 97 matching source files
across the launcher overlay, pinned clone and fork checkout.

These are expert subsystem checks. The experimental block and frozen-index state support below are implemented;
full block numerical qualification and draft-three target verification are
still required. The normal model/API does not yet perform speculative decoding.

## Experimental MTP block and frozen sparse drafts

The shared text MTP block now fuses the successor-token embedding with the
preceding target HC state, runs its attention/HC/block-FP8 MoE, and returns both
the collapsed hidden vector and all 10,240 HC values for the next proposal.
It borrows the target embedding, LM head and page reader. Its derived immutable
configuration has one full-attention layer and no PLE. This block is available
through `Qwen4CpuEngine.newMtpBlock`; the normal API still uses target-only decoding.

Native input fusion normalizes all HC branches together, runs two BF16 GEMMs,
and adds the embedding projection to each branch on CUDA. CUDA 13 NVRTC and
Driver API bindings implement the staged SM120 CuTe normalization geometry,
ordered local sums, ascending warp reductions and explicit approximate rsqrt.
CuBLAS plans stay with their caller thread and retain a 32 MiB workspace.
`gpu-mtp-fusion-parity.json` checks batches 1, 3, 33 and 128 against the actual
staged normalizer and PyTorch BF16 projections: all four outputs are exact,
including replay and another caller thread. Native modules and tracked buffers
return to zero. The peak stays within the 128 MiB fusion cache budget.

Teacher extension captures its final sparse selection. Three draft calls reuse
that selection and append only newly drafted causal positions; indexer projection,
compression and sparse scoring are skipped. Extra selection capacity includes
three tail entries. All request metadata and QSA prefixes support snapshot,
accepted-prefix restoration, reset and failure rollback. Indexed extension after
an unrestored draft prefix is refused. State snapshots reject wrong owners,
released checkpoints, future prefixes and pre-reset generations.

`gpu-qsa-draft-boundaries.json` captures 2,051 selected entries at position 2,063,
then drafts across a compression/page boundary with capacity 2,054. Constant-V
attention, selection preservation and rollback/replay are exact; no KV or draft
indexer results are downloaded. It also checks overflow, invalid future indices,
foreign/disposed/stale checkpoints and zero resources after close.
`gpu-qsa-mtp-regression.json` repeats the existing 12-step real-layer fresh-prefill
check with the extended QSA interface and passes its original tolerance.

Full MTP numerical qualification remains **incomplete**.
`gpu-mtp-block-unqualified.json` records five synthetic target-HC teacher inputs
and three chained drafts using real checkpoint weights. All four checked greedy
IDs match the staged kernels, and native replay, reset, injected-failure retry,
fourth-step refusal and resource release pass. The independent activation
comparison still fails 45 stage checks at rtol 0.016 / atol 0.004. The staged
reference itself replays exactly in `mtp-reference-replay.json`. Optional stage
traces identify HC/attention differences that propagate through the block.
An experimental grouped CUDA HC normalizer is available through the layer
executor's fourth constructor argument; the normal target provider keeps its
previous normalization path while MTP qualification continues.

`mtp-regression-tests.json` records 810 Java tests (no failures/errors, three
skips), 112 launcher tests and 106 matching source files.
`gpu-api-mtp-regression.json` repeats the three-caller normal CUDA API case and
reset/replay; all four results remain `[17, 478, 220]`, resources return to zero,
and the run takes about 267 seconds. This remains a short 4,096-capacity test.

`gpu-mtp-all-tensorcore-scores.json` records the latest synthetic block check
with the FP32 HC projections and tensor-core QSA scores. The four checked greedy
IDs still match and rollback/reset/failure retry and native resource release
pass. Activation failures fell from 45 to 20 stage checks; the original
rtol 0.016 / atol 0.004 gate still fails. This report preserves that failure.

`hc-qsa-regression-tests.json` records 810 Java tests (zero failures/errors,
three skips), 112 launcher tests and 109 matching source files.
`gpu-api-hc-qsa-regression.json` repeats the three-caller normal CUDA API check
with the new HC/QSA projections: all callers and reset/replay emit
`[17, 478, 220]`, with no checkpoint descriptors or native contexts/plans after
close. It executes 288,072 BF16/FP32 GEMMs and takes about
452 seconds, versus about 267 seconds in the preceding report.
The partitioned small-row path improves accumulation fidelity; this sequential
implementation still needs batching and fusion for throughput.

The MTP verifier now defaults to the actual staged FlashInfer paged-decode API,
which selects XQA on SM120, with BF16 queries and gathered FP8-rounded KV.
`--decode-reference pytorch` retains the earlier FP32 reference as an optional
diagnostic. `mtp-staged-reference-replay.json` confirms the new reference replays
exactly. Its current full block gate still fails during the draft trajectory;
changing the reference does not qualify speculative generation.

`gpu-qsa-xqa-c-api-feasibility.json` calls the exported XQA C ABI directly on
Torch-owned CUDA buffers. Both 6 and 2,054 selected-token cases match the public
FlashInfer API exactly with a 16 MiB workspace; the report records the module
hash and FlashInfer version. This establishes the native entry point, buffer
layout and stream-context mechanism for the Java/TornadoVM binding now described
below.

`gpu-mtp-native-combine-staged.json` records the current staged-kernel gate:
all five synthetic target-HC teacher inputs using real checkpoint weights pass;
20 draft-stage checks still fail at the unchanged rtol 0.016 / atol 0.004.
The four checked greedy IDs, rollback/reset/failure retry and resource release
still pass. `native-combine-regression-tests.json` records 810 Java tests (zero
failures/errors, three skips), 112 launcher tests and 109 matching source files.
`gpu-api-native-combine-regression.json` preserves all three normal CUDA API
callers and reset/replay with `[17, 478, 220]` and zero tracked native resources
and checkpoint descriptors after close. It takes about 450 seconds.

Native Java XQA decode is now implemented through TornadoVM library tasks and
TVM-FFI's exported C ABI. It uses packed BF16 queries and gathered FP8-rounded KV,
64-token pages, a static page table and a model-owned 16 MiB workspace per native
context. The provider saves and restores the caller's stream context, validates
prepared library hashes, and releases its workspace and library references at
plan close. The normal CUDA target model now selects this decode path; initial
fresh prefill keeps the existing staged sparse-prefill calculation. Optional
decoder score/probability traces are labelled as reference intermediates because
XQA owns its internal softmax.

The shared MTP expert's three BF16 projections now use native cuBLAS GEMMs with
one 32 MiB workspace per layer. The workspace is rebound after every caller
stream change. Routed expert padding is reduced to 16 rows, keeping the complete
MoE within its existing 128 MiB budget.

`gpu-xqa-portable-boundary-parity.json` checks 1, 6, 63, 64, 65, 2,048, 2,051 and
2,054 selected tokens against the actual staged FlashInfer API. All 16 cases
across two library reloads are exact, including graph replay, another caller
thread and zero native workspaces after close. `gpu-fp8-shared-gemm-thread-parity.json`
repeats the three real FP8 MoE cases with the original tolerance, eviction,
caller-thread replay and resource checks.

`gpu-mtp-portable-native-xqa-parity.json` now **passes** all activation-stage
checks for five synthetic teacher inputs and three chained drafts at the
original rtol 0.016 / atol 0.004. The four checked greedy IDs match. Reset,
rollback, injected-failure retry, frozen selection and native resource release
pass; KV is never downloaded. The older failing reports above preserve the
preceding diagnostics. Real target-derived draft trajectories and speculative
API generation still require qualification.

Prepare the pinned native dependencies with the CUDA Python environment:

```bash
python3 scripts/flash-next/prepare_jitllm_qsa.py
```

This requires FlashInfer 0.6.18, TVM-FFI 0.1.11, CMake, Ninja and a C++ compiler.
The preparer builds TVM-FFI at `3c35034fd1026011736e19a4e0e1ed0f22058c42`
with `TVM_FFI_BACKTRACE_ON_SEGFAULT=OFF`; the packaged library installs a signal
handler that conflicts with HotSpot. It records source and binary provenance in
`.local/jitllm-native-libs/qsa-provenance.json`, which the runtime wrapper verifies
before exporting library paths and hashes. Set `JITLLM_QSA_PROVENANCE` to reuse a
prepared manifest from another checkout. The binding currently requires SM120
and the checkpoint's 24-query-head / two-KV-head / 256-dimension geometry. CUDA
graph capture remains explicitly unqualified.

`gpu-api-xqa-regression.json` now runs XQA in the normal CUDA target model.
All three callers and reset/replay still emit `[17, 478, 220]`. The run executes
96 XQA decodes and releases all XQA workspaces, native contexts/plans and
checkpoint descriptors at close. It takes about 481 seconds in the sequential
pipeline; efficient batching and layer-to-layer fusion remain required.

`xqa-mtp-regression-tests.json` records 810 Java tests (zero failures/errors,
three skips), 112 launcher tests and 113 matching source files. It distinguishes
passing native block checks from the still-missing real target-derived MTP
qualification, acceptance loop and full-context concurrent capacity checks.

Target verification/acceptance, rejection correction, stochastic sampling,
real target-derived draft trajectories and batched verification are still
required. The block checks do not establish working speculative API generation.

## Normal CUDA model/API integration

`LocalModels.load(checkpoint, ModelOptions.builder().backend(BackendId.CUDA).build())`
now loads the native Flash-Next executors. A session owns its DeltaNet, PLE and
QSA state; the model owns projection/HC/MoE workspaces and the bounded packed
expert cache. Simultaneous session callers serialize device execution on those
shared workspaces. This is admission and isolation, not batched GPU scheduling.
`prepare()` reports the actual CUDA/FP8 path without advancing the sequence.
Explicit unsupported storage or execution-policy requests fail and release the
loaded model. With no explicit storage override, Flash-Next selects physical
FP8 E4M3 KV pages. `StorageOptions.fp8()` requests that format explicitly; legacy
models reject it when they have no FP8 writer/reader. A pending output token is
ingested on the next turn. Initial prompts no longer lose one output token to a
misidentified header seed.

CUDA requires the CUDA 13 native libraries and the same pinned TornadoVM SDK.
The provider uses the process-resolved default device unless `jitllm.qwen4.device`
selects a CUDA backend ordinal. The checkpoint runner sets this property without
reordering TornadoVM's device list. Device caches use
`jitllm.qwen4.gpuCacheBytes` (default half of device VRAM minus 4 GiB,
capped at 16 GiB: about 12 GiB on this 32 GiB RTX 5090). `gpuProjectionBytes` reserves
up to 8 GiB of that for dense projections, with the remainder assigned to MoE;
it may be explicitly overridden. The other limits are `gpuStateBytes` (4 GiB), `gpuLayerBytes` (64 MiB), `gpuQsaBytes`
(16 GiB of full-context logical reservation; pages materialize as reached),
`checkpointCapacity` (8), and `pageCacheBytes` (1 MiB), all under the
`jitllm.qwen4.` property prefix. The BF16 GEMM cache is separately capped at
256 MiB. These independent limits are not a complete physical-memory prediction.
Full native context remains 262,144; the short API qualification uses a 4,096-token
capacity. Long-context concurrent physical memory and scheduling remain unqualified.

The tokenizer merge map stays private behind an unmodifiable `HashMap` view.
The old `Map.copyOf` open-addressing construction spent over a minute handling
colliding long token-pair keys during an observed API load. Tokenization remains
checked against the checkpoint's Hugging Face tokenizer. Preflight and rejected
API loads now release any opened model resources.

Run the normal API probe with:

```bash
python3 scripts/flash-next/jitllm_runtime.py gpu-api-probe /path/to/Qwen3.8-Flash-Next-NVFP4 scripts/flash-next/examples/short-chatml-gpu-api.json
```

This launches three simultaneous API callers, checks token events/positions,
refuses a fourth session and model close with live sessions, then resets and
replays one session. It also checks explicit FP16 refusal, native execution and
checkpoint/native resource release. `gpu-api-conformance.json` records three callers and a reset/replay case, all
exact against a fresh uncached staged `[17, 478, 220]` result. They ingest the
16-token prompt and emit three tokens at API position 19, run 69,120 native
expert GEMMs, and release checkpoint file descriptors, native contexts,
descriptors and expert registrations to zero. The run took about 264 seconds.
One-second GPU samples peaked at 29,065 MiB used, including the 4,395 MiB desktop
baseline, and retained at least 3,044 MiB free. GPU usage returned to baseline
after process exit. The earlier 16 GiB weight-cache setting failed on the third
call and is preserved in `gpu-api-memory-failure.json`; the adaptive default
keeps the same three callers live while paging more experts. This measures one
short prompt with 4,096-token capacity, not full-context concurrent workloads.
`gpu-api-regression-tests.json` records 806 Java tests (zero failures/errors,
three skipped), 112 launcher tests, 247 exact tokenizer cases and 88 matching
source files. Layer-to-layer activation fusion, efficient batched prefill, MTP,
vision and complete chat replay remain required.

## Build and reproduce checks

Run commands from the root of this JITLLM checkout. Use JDK 25 and the pinned CUDA
TornadoVM SDK above. Export `JAVA_HOME` for JDK 25 and `TORNADOVM_HOME` for an
existing SDK; the normal upstream Maven build remains available:

```bash
./mvnw -B -Djitllm.VectorBitSize=256 clean test
./mvnw -B -DskipTests package
python3 -m unittest discover -s scripts/flash-next -p 'test_*.py' -v
```

The checkpoint fixture suite compiles only the storage-reader Java subset and
requires a JDK, with no checkpoint, Python packages, TornadoVM or GPU needed.
The current upstream suite runs 803 tests with zero failures/errors and three
skipped; the original launcher suite ran 112 tests. The 14 portable checkpoint
fixture tests are included in this fork. `regression-tests.json` preserves the
original commands and counts as historical evidence. The JITLLM test command
selects 256-bit CPU vectors because the upstream Q4_1 dot-product implementation
rejects this host's default 512-bit vector species.
`recurrent-regression-tests.json` records the current suites and GPU recurrence
checks. Recurrent memory estimates now count the actual float backing storage of
convolution and PLE state rather than their BF16-rounded values' two-byte format.

If an SDK is not installed, this optional helper invokes JITLLM's upstream SDK
builder at the pinned revision. It caches or downloads a JDK 21 JVMCI image
unless `JVMCI_SOURCE_JDK` points to one. `JAVA_HOME`/PATH must select JDK 25;
CUDA/cuDNN compiler headers and libraries must be available. For pip-installed
CUDA 13 and cuDNN 9, export `JITLLM_NVIDIA_DIR` as their containing `nvidia`
directory (the directory containing `cu13` and `cudnn`). Otherwise the helper
uses the system compiler/linker environment. Private unversioned aliases for
CUDA 13/cuBLAS/cuDNN let the pinned SDK bind the same vendor libraries as the
reference environment; occupied or redirected alias paths are rejected.

```bash
python3 scripts/flash-next/prepare_jitllm_runtime.py
python3 scripts/flash-next/jitllm_runtime.py build
python3 scripts/flash-next/jitllm_runtime.py devices
python3 scripts/flash-next/jitllm_runtime.py test
```

The wrapper accepts `TORNADOVM_HOME`, `JITLLM_TORNADOVM_INSTALL`, or `--install`
(the latter two point to an upstream builder installation with `provenance.json`).
A direct SDK takes precedence. No checkpoint files are downloaded by these tools.
Use the local snapshot of `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, for the recorded comparisons.

Verification scripts require Python with PyTorch and, depending on the script,
`safetensors`, `tokenizers`, FlashInfer and the staged Qwen4-capable SGLang source.
SGLang-specific layer checks import its actual CUDA kernels; an unrelated stock
SGLang release may not contain those modules. They are optional independent
oracles, separate from the native JITLLM execution. Use the staged environment
for those scripts; its package/revision details are recorded in
`benchmarks/jitllm-flash-next/reference-environment.json`.

```bash
python scripts/flash-next/verify_jitllm_checkpoint.py MODEL_DIR
python scripts/flash-next/verify_jitllm_gpu.py MODEL_DIR --device 1
python scripts/flash-next/verify_jitllm_metadata.py MODEL_DIR
python scripts/flash-next/verify_jitllm_layers.py MODEL_DIR
python scripts/flash-next/verify_jitllm_deltanet.py MODEL_DIR
python scripts/flash-next/verify_jitllm_layers.py MODEL_DIR --gpu-hc-ple --gpu-projections --fused-ple-decode --hc-rows 4096
python scripts/flash-next/verify_jitllm_bf16_batch.py MODEL_DIR
python scripts/flash-next/verify_jitllm_deltanet.py MODEL_DIR --gpu-recurrence --gpu-projections --device 1
python scripts/flash-next/verify_jitllm_qsa.py MODEL_DIR --fp8-kv --fused-precision --fresh-prefill
python scripts/flash-next/verify_jitllm_activation.py MODEL_DIR
python scripts/flash-next/verify_jitllm_moe.py MODEL_DIR
```

Each verifier supports `--output FILE.json`. The strict MoE verifier is expected
to report the outstanding numeric failure. Some reference scripts choose CUDA
device 1; adjust to the reference environment's GPU enumeration when necessary.
TornadoVM ordinals differ from `nvidia-smi` on the test host: Tornado `0:0` is the
RTX PRO 6000 and `0:1` is the RTX 5090. Inspect `devices` on another host.

## Run the implemented paths

After building, the wrapper exposes these diagnostic paths:

```bash
python3 scripts/flash-next/jitllm_runtime.py api-probe MODEL_DIR
python3 scripts/flash-next/jitllm_runtime.py model-probe MODEL_DIR REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py cache-probe MODEL_DIR CACHE_REQUEST.json --device 1
python3 scripts/flash-next/jitllm_runtime.py recurrent-probe MODEL_DIR RECURRENT_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py hc-ple-probe MODEL_DIR HC_PLE_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py bf16-batch-probe MODEL_DIR BF16_BATCH_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py nvfp4-batch-probe MODEL_DIR NVFP4_BATCH_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py fp8-batch-probe MODEL_DIR FP8_BATCH_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py fp8-moe-probe MODEL_DIR FP8_MOE_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py mtp-fusion-probe MODEL_DIR MTP_FUSION_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py mtp-block-probe MODEL_DIR MTP_BLOCK_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py qsa-draft-probe --device 1
python3 scripts/flash-next/jitllm_runtime.py moe-probe MODEL_DIR MOE_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py qsa-probe MODEL_DIR QSA_REQUEST.json
python3 scripts/flash-next/jitllm_runtime.py qsa-paging-probe --device 1
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-gpu-recurrence.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-gpu-rollback.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-hc-ple-rollback.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-hc-ple-bf-prefill.json
```

The included example reproduces the 16-token prompt and requests three greedy
response tokens. Set its `device` ordinal for the local GPU. The recorded result
is `[17, 478, 220]` (`2 + `), matching the uncached staged reference. Native
execution took about 196 seconds on the recorded host; these scalar projection
kernels are a correctness implementation, not a production performance result.

A raw generation request can contain `input_tokens`, `max_tokens`, and
`trace_layers`. Setting `gpu_projections: true`, `device: 1`, and
`gpu_cache_bytes: 17179869184` selects the GPU projection cache. Attention,
recurrent updates, PLE gating, routing and other nonlinear operations still run
on the host unless `gpu_recurrence: true` selects CUDA DeltaNet. Its
`checkpoint_capacity` defaults to eight per layer/sequence, and `gpu_state_bytes`
defaults to 4 GiB across all live recurrent sessions and checkpoint banks.
Setting `gpu_hc_ple: true` also selects CUDA HC and PLE arithmetic.
Its `gpu_layer_bytes` budget defaults to 64 MiB for model HC buffers and
request PLE buffers/checkpoints. Setting `gpu_qsa: true` selects CUDA sparse
attention with physical FP8 KV pages. Its `gpu_qsa_bytes` reservation budget
defaults to 4 GiB across all live QSA sessions; snapshots use the shared
`checkpoint_capacity`. MoE nonlinear work remains on the host.
This is an experimental diagnostic path, not the full GPU model
provider. `cache-probe` takes a `matrices` list of checkpoint projection prefixes
and an optional `cache_bytes` budget.

## Remaining requirements for the original goal

1. Broaden text conformance beyond the successful short ChatML case. Broaden
   the native MoE numeric checks, validate prompt/decode and reused-session
   trajectories, and preserve the historical failure record. Qualify the staged precision policy in full-model prefill and reused-session trajectories.
2. Complete GPU QSA provider/long-context qualification and fuse the native layer plans,
   including grouped native NVFP4 projections and complete FP8 draft-layer integration. The normal CUDA
   model provider now dispatches through its owned plans; production serving remains to be qualified.
3. Qualify the MTP/NextN block and implement draft-three target verification,
   rejection correction and accepted-prefix rollback across target and draft states.
4. Qualify physical FP8 KV page transitions and efficient 262,144-token sparse indexing,
   prefill/chunking, memory planning and three-request scheduling. Configuration
   and admission checks do not prove full-context/concurrent inference.
5. Implement native vision/image processing and exact checkpoint chat-template
   behavior for replayed reasoning and tools; verify their API requests.
6. Add launcher routing and systemd deployment after correct native serving is
   demonstrated. Verify real text, long-context retrieval, NVMe page activity,
   bounded memory, MTP, images and tools against current runtime evidence.

The original goal remains active. No native production route has been published.

# JITLLM Flash-Next implementation

The goal is native JITLLM inference and serving of
`nvidia/Qwen3.8-Flash-Next-NVFP4`, with bounded NVMe PLE/Engram staging and the
features of the original launcher's `qwen-flash-next-staged` profile. The full 48-layer text pipeline now executes on the host and with native GPU
weight projections and DeltaNet state updates. The recorded 16-token ChatML
comparison now matches three greedy staged-reference tokens exactly; broader
full-model correctness remains to be qualified. The normal JITLLM loader/API can load the
checkpoint on CPU; production GPU serving is still under implementation.

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
  These are scalar reference kernels; tensor-core execution remains required.
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
  FlashInfer across four global scales. The strict MoE output gate remains red.
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
The current upstream suite runs 799 tests with zero failures/errors and three
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
uses the system compiler/linker environment.

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
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-gpu-recurrence.json
python3 scripts/flash-next/jitllm_runtime.py generate MODEL_DIR scripts/flash-next/examples/short-chatml-gpu-rollback.json
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
HC, PLE, QSA and MoE nonlinear work still runs on the host. This is an experimental diagnostic path, not the full GPU model
provider. `cache-probe` takes a `matrices` list of checkpoint projection prefixes
and an optional `cache_bytes` budget.

## Remaining requirements for the original goal

1. Broaden text conformance beyond the successful short ChatML case. Resolve
   the strict MoE numeric differences, validate prompt/decode and reused-session
   trajectories, and preserve the historical failure record. Long prefill also
   needs the staged HC projection precision policy.
2. Implement full GPU hyper-connection, PLE, QSA and MoE execution,
   including efficient packed NVFP4 tensor-core projections. Connect that plan
   to the normal model provider; it currently refuses full GPU loads explicitly.
3. Add MTP/NextN draft-three verification, block-FP8 draft layers, QSA index
   reuse, and rejection/accepted-prefix rollback for every request state.
4. Implement physical FP8 KV storage, efficient 262,144-token sparse indexing,
   prefill/chunking, memory planning and three-request scheduling. Configuration
   and admission checks do not prove full-context/concurrent inference.
5. Implement native vision/image processing and exact checkpoint chat-template
   behavior for replayed reasoning and tools; verify their API requests.
6. Add launcher routing and systemd deployment after correct native serving is
   demonstrated. Verify real text, long-context retrieval, NVMe page activity,
   bounded memory, MTP, images and tools against current runtime evidence.

This branch is experimental. No native production route has been published.

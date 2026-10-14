"""Compile the pinned staged SM120 HC mix for direct CUDA-driver execution in JITLLM."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from jitllm_runtime import ROOT

SGLANG_REF = "745de73ba3c136b6f99b7a3e2177ed1a8eef4a56"
SOURCE_PATH = "python/sglang/srt/layers/hc_mix_triton.py"
ABI = ["u64"] * 6 + ["u32"] * 4 + ["f32"] + ["u64"] * 2


def prepare(output, device=None):
    import torch
    import triton
    from sglang.srt.layers import hc_mix_triton

    source = Path(hc_mix_triton.__file__).resolve()
    revision = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(["git", "-C", str(source.parent), "show", f"{SGLANG_REF}:{SOURCE_PATH}"])
    if revision != SGLANG_REF or source.read_bytes() != expected:
        raise RuntimeError("HC mix must match the staged SGLang revision and unmodified kernel source")
    if device is None:
        device = next((i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)), torch.cuda.current_device())
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("The native HC binding currently requires SM120")
    rows, columns, lowrank, dimension, branches = 1, 10240, 320, 2560, 4
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    tensors = [torch.zeros((rows, columns), device="cuda", dtype=torch.bfloat16),
               torch.zeros((lowrank, columns), device="cuda", dtype=torch.bfloat16),
               torch.zeros((columns, lowrank), device="cuda", dtype=torch.bfloat16),
               torch.empty((16, lowrank), device="cuda", dtype=torch.float32),
               torch.empty((rows, dimension), device="cuda", dtype=torch.bfloat16),
               torch.zeros(3, device="cuda", dtype=torch.int32)]
    kernel = hc_mix_triton._hc_mix_persistent_kernel.warmup(
        *tensors, columns, lowrank, dimension, rows, sms, 1 / branches,
        ROWS=16, HC=branches, BLOCK_N=32, BLOCK_K=256, BLOCK_J=32, BLOCK_R=64,
        num_warps=8, grid=(sms,))
    entry = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", kernel.asm["ptx"], re.S)
    if not entry or re.findall(r"\.param\s+\.(\w+)", entry[2]) != ABI:
        raise RuntimeError("The Triton HC CUDA ABI changed; refuse to emit an incompatible binary")
    metadata = kernel.metadata
    if metadata.num_warps != 8 or metadata.global_scratch_size or metadata.profile_scratch_size or metadata.shared != 51200:
        raise RuntimeError("The qualified HC launch geometry changed")
    # Run and replay once to verify barrier reset before exporting the binary.
    for _ in range(2):
        hc_mix_triton._hc_mix_persistent_kernel[(sms,)](
            *tensors, columns, lowrank, dimension, rows, sms, 1 / branches,
            ROWS=16, HC=branches, BLOCK_N=32, BLOCK_K=256, BLOCK_J=32, BLOCK_R=64, num_warps=8)
        torch.cuda.synchronize()
        if torch.count_nonzero(tensors[4]).item() or torch.count_nonzero(tensors[5]).item():
            raise RuntimeError("HC kernel failed its zero-input/barrier replay check")
    output.parent.mkdir(parents=True, exist_ok=True)
    cubin = output.parent / "hc-mix-sm120.cubin"
    temporary = cubin.with_suffix(".tmp")
    temporary.write_bytes(kernel.asm["cubin"]); temporary.replace(cubin)
    report = {"sglang_ref": revision, "source_sha256": hashlib.sha256(expected).hexdigest(),
              "triton_version": triton.__version__, "torch_version": torch.__version__,
              "compute_capability": [12, 0], "kernel": entry[1], "abi": ABI,
              "cubin": str(cubin.resolve()), "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
              "rows": rows, "columns": columns, "lowrank": lowrank, "dimension": dimension, "branches": branches,
              "num_warps": metadata.num_warps, "shared_bytes": metadata.shared,
              "global_scratch_bytes": 0, "profile_scratch_bytes": 0, "barrier_replay_passed": True}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n"); temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/jitllm-native-libs/hc-provenance.json")
    parser.add_argument("--device", type=int, help="CUDA/PyTorch ordinal")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.absolute(), args.device), indent=2))

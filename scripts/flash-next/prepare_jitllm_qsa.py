"""Prepare the pinned XQA module and a TVM-FFI build that preserves HotSpot's signal handler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from jitllm_runtime import ROOT

TVM_FFI_REF = "3c35034fd1026011736e19a4e0e1ed0f22058c42"
FLASHINFER_VERSION = "0.6.18"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, build, output, device=None):
    import torch
    import flashinfer
    import tvm_ffi
    from flashinfer.jit.xqa import gen_xqa_module

    if flashinfer.__version__ != FLASHINFER_VERSION or tvm_ffi.__version__ != "0.1.11":
        raise RuntimeError("Native QSA currently requires FlashInfer 0.6.18 and TVM-FFI 0.1.11")
    if device is None:
        device = next((i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)), torch.cuda.current_device())
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("The Flash-Next XQA binding currently requires SM120")
    if not source.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.1.11", "https://github.com/apache/tvm-ffi", str(source)], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != TVM_FFI_REF or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("TVM-FFI source must be clean at the pinned revision; local edits are preserved")
    subprocess.run(["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1", "3rdparty/dlpack"], check=True)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    "-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF", "-DTVM_FFI_USE_LIBBACKTRACE=OFF",
                    "-DTVM_FFI_BUILD_TESTS=OFF", "-DTVM_FFI_BUILD_PYTHON_MODULE=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "tvm_ffi_shared", "-j", "8"], check=True)
    ffi = (build / "lib/libtvm_ffi.so").resolve()
    spec = gen_xqa_module(torch.bfloat16, torch.bfloat16, 64, 256, 12, False, torch.bfloat16, 1, False)
    spec.build_and_load()
    xqa = Path(spec.get_library_path()).resolve()
    report = {"tvm_ffi_ref": revision, "tvm_ffi_version": "0.1.11", "backtrace_on_segfault": False,
              "flashinfer_version": flashinfer.__version__, "compute_capability": [12, 0],
              "tvm_ffi_library": str(ffi), "tvm_ffi_sha256": sha(ffi), "xqa_library": str(xqa), "xqa_sha256": sha(xqa),
              "workspace_bytes": 16777216, "head_dim": 256, "query_heads": 24, "kv_heads": 2, "page_size": 64}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".local/jitllm-tvm-ffi-src")
    parser.add_argument("--build", type=Path, default=ROOT / ".local/jitllm-tvm-ffi-jvm-build")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/jitllm-native-libs/qsa-provenance.json")
    parser.add_argument("--device", type=int, help="CUDA/PyTorch ordinal, independent of TornadoVM's device ordinal")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source.absolute(), args.build.absolute(), args.output.absolute(), args.device), indent=2))

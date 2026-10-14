"""Private unversioned CUDA aliases for the pinned SDK's FFM library bindings."""
from pathlib import Path
import hashlib
import json


def native_hc_environment(provenance: Path, env: dict) -> None:
    if not provenance.is_file():
        return
    data = json.loads(provenance.read_text())
    if (data.get("sglang_ref") != "745de73ba3c136b6f99b7a3e2177ed1a8eef4a56"
            or data.get("compute_capability") != [12, 0] or data.get("rows") != 1
            or data.get("columns") != 10240 or data.get("lowrank") != 320
            or data.get("dimension") != 2560 or data.get("branches") != 4
            or data.get("kernel") != "_hc_mix_persistent_kernel"
            or data.get("shared_bytes") != 51200 or data.get("num_warps") != 8
            or data.get("abi") != ["u64"] * 6 + ["u32"] * 4 + ["f32"] + ["u64"] * 2
            or data.get("global_scratch_bytes") != 0 or data.get("profile_scratch_bytes") != 0
            or data.get("barrier_replay_passed") is not True):
        raise RuntimeError("Prepared HC kernel has an incompatible source/launch ABI")
    path = Path(data["cubin"])
    digest = data["cubin_sha256"]
    if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Prepared HC kernel changed: {path}")
    env["JITLLM_HC_CUBIN"] = str(path)
    env["JITLLM_HC_SHA256"] = digest


def native_qsa_environment(provenance: Path, env: dict) -> None:
    if not provenance.is_file():
        return
    data = json.loads(provenance.read_text())
    if data.get("backtrace_on_segfault") is not False or data.get("tvm_ffi_ref") != "3c35034fd1026011736e19a4e0e1ed0f22058c42":
        raise RuntimeError("Native QSA needs the pinned TVM-FFI build with SIGSEGV handling disabled")
    for prefix, variable in [("tvm_ffi", "JITLLM_TVM_FFI"), ("xqa", "JITLLM_FLASHINFER_XQA")]:
        path = Path(data[prefix + "_library"])
        digest = data[prefix + "_sha256"]
        if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Prepared native QSA library changed: {path}")
        env[variable + "_LIBRARY"] = str(path)
        env[variable + "_SHA256"] = digest


def prepare_native_aliases(native: Path, directory: Path) -> Path:
    native = native.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    origins = {"libcudnn.so": native / "cudnn/lib/libcudnn.so.9",
               "libcudart.so": native / "cu13/lib/libcudart.so.13",
               "libnvrtc.so": native / "cu13/lib/libnvrtc.so.13",
               "libcublas.so": native / "cu13/lib/libcublas.so.13",
               "libcublasLt.so": native / "cu13/lib/libcublasLt.so.13"}
    # Preflight every path before making changes; occupied paths belong to the caller.
    for name, origin in origins.items():
        link = directory / name
        if not origin.is_file():
            raise RuntimeError(f"Missing native CUDA dependency: {origin}")
        if link.is_symlink():
            if link.resolve() != origin.resolve():
                raise RuntimeError(f"Native alias points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Native alias path is occupied: {link}")
    for name, origin in origins.items():
        link = directory / name
        if not link.is_symlink():
            link.symlink_to(origin)
    return directory

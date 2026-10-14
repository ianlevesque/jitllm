"""Execute packed target and MTP weight projections on an actual TornadoVM CUDA device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from jitllm_runtime import ROOT, INSTALL
from prepare_jitllm_source import JITLLM_REF


def verify(model: Path, device: int) -> dict:
    reports = []
    for layer in ["model.language_model.layers.0", "mtp.layers.0"]:
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            matrix = layer + ".mlp.experts.0." + projection
            result = subprocess.run([sys.executable, str(ROOT / "jitllm_runtime.py"), "probe", str(model),
                                     "--matrix", matrix, "--device", str(device)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            lines = [line for line in result.stdout.splitlines() if line.startswith('{"')]
            if len(lines) != 1:
                raise RuntimeError("Expected one GPU probe result: " + result.stdout)
            report = json.loads(lines[0])
            if report["gpu_kernel_ns"] <= 0:
                raise AssertionError("Missing GPU kernel execution")
            reports.append(report)
    provenance = json.loads((INSTALL / "provenance.json").read_text())
    return {"jitllm_ref": JITLLM_REF, "tornadovm_ref": provenance["ref"], "checkpoint": str(model.resolve()),
            "activation_dtype": "F32", "weight_formats": ["NVFP4", "block-FP8"], "tornado_device_index": device, "projections": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.model, args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)

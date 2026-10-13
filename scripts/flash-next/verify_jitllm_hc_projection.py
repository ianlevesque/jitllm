"""Compare real native HC BF16/FP32 projections with CUDA tensor-core accumulation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent


def verify(model, matrices, device=1):
    torch.set_num_threads(4)
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    cases = []
    for name in matrices:
        with safe_open(model / mapping[name + ".weight"], framework="pt", device="cpu") as file:
            weight = file.get_tensor(name + ".weight").cuda()
        columns = weight.shape[1]
        partition = 256 if name.endswith("input_mix_weight_down") else columns
        vector = torch.arange(columns, device="cuda", dtype=torch.float32)
        for step in range(3):
            values = (.4 * torch.sin(vector * .013 + step * .27)).bfloat16().reshape(1, -1)
            batch = values.repeat(16, 1)
            expected = None
            for start in range(0, columns, partition):
                part = torch.mm(batch[:, start:start + partition], weight[:, start:start + partition].T, out_dtype=torch.float32)[0]
                expected = part if expected is None else expected + part
            expected = expected.cpu()
            with tempfile.TemporaryDirectory(prefix="jitllm-hc-projection-") as directory:
                request = Path(directory) / "request.json"
                request.write_text(json.dumps({"matrix": name, "partition": partition, "device": device, "input": values.float().cpu().flatten().tolist()}))
                result = subprocess.run([sys.executable, str(SCRIPT_DIR / "jitllm_runtime.py"), "hc-projection-probe", str(model), str(request)], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr + "\n" + result.stdout)
                actual = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"')))
            observed = torch.tensor(actual.pop("output")); difference = observed - expected
            actual.update({"step": step, "max_absolute_error": difference.abs().max().item(), "exact_fraction": (difference == 0).float().mean().item()})
            print(actual, flush=True)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            assert actual["replay_exact"] and actual["caller_thread_exact"] and actual["resources_after_close"] == 0
            cases.append(actual)
    return {"passed": True, "checkpoint": str(model), "reference": "CUDA BF16 tensor-core dot products with FP32 output and ordered K-partition sum", "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("model", type=Path); parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--matrices", nargs="+", default=[prefix + "." + projection for prefix in ("mtp.layers.0.attn_hyper_connection", "model.language_model.layers.1.attn_hyper_connection") for projection in ("input_mix_weight_down", "input_mix_weight_up")])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = json.dumps(verify(args.model, args.matrices, args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(report + "\n")
    print(report)

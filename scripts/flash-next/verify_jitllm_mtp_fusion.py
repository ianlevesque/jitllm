"""Compare native MTP global normalization and BF16 fusion with staged CUDA kernels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sgl_kernel import gemma_rmsnorm

SCRIPT_DIR = Path(__file__).resolve().parent


def verify(model, batches, device=1):
    torch.set_num_threads(4)
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model / "config.json").read_text())["text_config"]

    def weight(name):
        name = "mtp." + name
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name).cuda()

    ew, hw, en, hn = [weight(name) for name in ["fc_embedding.weight", "fc_hidden.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"]]
    dim, branches = config["hidden_size"], config["hc_count"]
    vector = torch.arange(dim, device="cuda", dtype=torch.float32)
    amplitudes = torch.arange(1, branches + 1, device="cuda", dtype=torch.float32).reshape(branches, 1) / 2
    cases = []
    for rows in batches:
        embeddings = torch.stack([(.4 * torch.sin(vector * .013 + row * .17)).bfloat16() for row in range(rows)])
        hidden = torch.stack([(amplitudes * torch.cos(vector * .021 + row * .11)).flatten().bfloat16() for row in range(rows)])
        normalized_e = gemma_rmsnorm(embeddings, en, config["rms_norm_eps"])
        normalized_h = gemma_rmsnorm(hidden, hn, config["rms_norm_eps"])
        expected = (F.linear(normalized_e, ew).unsqueeze(1)
                    + F.linear(normalized_h.reshape(rows, branches, dim), hw)).float().cpu().flatten()
        with tempfile.TemporaryDirectory(prefix="jitllm-mtp-fusion-") as directory:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps({"rows": rows, "device": device, "embeddings": embeddings.float().cpu().flatten().tolist(),
                                           "hidden": hidden.float().cpu().flatten().tolist()}))
            process = subprocess.run([sys.executable, str(SCRIPT_DIR / "jitllm_runtime.py"), "mtp-fusion-probe", str(model), str(request)],
                                     capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stderr + "\n" + process.stdout)
            actual = json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{"')))
        observed = torch.tensor(actual.pop("output")); difference = observed - expected
        actual.update({"max_absolute_error": difference.abs().max().item(), "exact_fraction": (difference == 0).float().mean().item()})
        print(actual, flush=True)
        torch.testing.assert_close(observed, expected, rtol=.008, atol=.0001)
        assert actual["replay_exact"] and actual["caller_thread_exact"] and actual["resident_bytes_after_close"] == 0
        cases.append(actual)
    return {"passed": True, "checkpoint": str(model), "reference": "Staged CUDA Gemma RMSNorm and PyTorch BF16 linear fusion",
            "hidden_norm_width": dim * branches, "rtol": .008, "atol": .0001, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path); parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 33, 128])
    parser.add_argument("--device", type=int, default=1); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = json.dumps(verify(args.model, args.batches, args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(report + "\n")
    print(report)

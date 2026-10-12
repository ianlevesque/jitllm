"""Compare distinct-row native block-FP8 GEMM with staged Triton W8A8 experts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import triton.language as tl
from safetensors import safe_open
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

SCRIPT_DIR = Path(__file__).resolve().parent


def verify(model, matrices, batches, device=1):
    torch.set_num_threads(4)
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    cases = []
    for prefix in matrices:
        with safe_open(model / mapping[prefix + ".weight"], framework="pt", device="cpu") as file:
            weight = file.get_tensor(prefix + ".weight")
        with safe_open(model / mapping[prefix + ".weight_scale_inv"], framework="pt", device="cpu") as file:
            scales = file.get_tensor(prefix + ".weight_scale_inv")
        columns, dimension = weight.shape[1], weight.shape[0]
        vector = torch.arange(columns, device="cuda", dtype=torch.float32)
        route = .113 if prefix.endswith("down_proj") else 1.
        for batch in batches:
            # Include the .54 phase that exposed fast-reciprocal encoding ties.
            x = torch.stack([(.4 * torch.sin(vector * .013 + .54 + row * .17)).bfloat16() for row in range(batch)])
            encoded, input_scales = sglang_per_token_group_quant_fp8(x, 128)
            padded_count = (batch + 15) // 16 * 16
            ids = torch.zeros((batch, 1), device="cuda", dtype=torch.int32)
            sorted_ids = torch.arange(padded_count, device="cuda", dtype=torch.int32).clamp_max(batch)
            experts = torch.zeros((padded_count // 16,), device="cuda", dtype=torch.int32)
            padded = torch.tensor([padded_count], device="cuda", dtype=torch.int32)
            routes = torch.full((batch, 1), route, device="cuda", dtype=torch.float32)
            output = torch.empty((batch, 1, dimension), device="cuda", dtype=torch.bfloat16)
            invoke_fused_moe_kernel(x, weight.cuda().unsqueeze(0), None, output, None,
                                   scales.cuda().unsqueeze(0), None, routes, ids, sorted_ids, experts, padded,
                                   route != 1., 1,
                                   {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                                    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
                                   tl.bfloat16, True, False, False, False, False, [128, 128])
            expected = output.float().cpu().flatten()
            with tempfile.TemporaryDirectory(prefix="jitllm-fp8-batch-") as directory:
                request = Path(directory) / "request.json"
                request.write_text(json.dumps({"matrix": prefix, "batch_rows": batch, "device": device,
                                               "route_weight": route, "input": x.float().cpu().flatten().tolist()}))
                process = subprocess.run([sys.executable, str(SCRIPT_DIR / "jitllm_runtime.py"), "fp8-batch-probe",
                                          str(model), str(request)], capture_output=True, text=True)
                if process.returncode:
                    raise RuntimeError(process.stderr + "\n" + process.stdout)
                reports = [line for line in process.stdout.splitlines() if line.startswith('{"')]
                if len(reports) != 1:
                    raise RuntimeError("Expected one block-FP8 report: " + process.stdout)
                actual = json.loads(reports[0])
            assert bytes.fromhex(actual.pop("encoded_hex")) == bytes(encoded.view(torch.uint8).cpu().flatten().tolist())
            torch.testing.assert_close(torch.tensor(actual.pop("input_scales")), input_scales.cpu().flatten(), rtol=0, atol=0)
            observed = torch.tensor(actual.pop("output"))
            torch.testing.assert_close(observed, expected, rtol=.008, atol=.0001)
            difference = observed - expected
            assert actual["native_calls"] > 0 and actual["kernel_ns"] > 0
            assert actual["caller_thread_exact"] and actual["replay_exact"]
            assert actual["resident_bytes_after_close"] == actual["native_contexts_after_close"] == actual["native_plans_after_close"] == 0
            actual.update({"activation_bytes_exact": True, "activation_scales_exact": True,
                           "max_absolute_error": difference.abs().max().item(),
                           "exact_fraction": (difference == 0).float().mean().item()})
            cases.append(actual)
            print(actual, flush=True)
    return {"passed": True, "checkpoint": str(model), "reference": "Staged Triton block-FP8 W8A8",
            "routing_weight_before_bf16_store": True, "rtol": .008, "atol": .0001, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--matrices", nargs="+", default=["mtp.layers.0.mlp.experts.0." + name for name in ("gate_proj", "up_proj", "down_proj")])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 33, 128])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.model, args.matrices, args.batches, args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)

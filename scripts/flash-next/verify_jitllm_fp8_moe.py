"""Compare native MTP block-FP8 MoE with the staged Triton W8A8 path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
import triton.language as tl
from safetensors import safe_open
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

ROOT = Path(__file__).resolve().parent
PREFIX = "mtp.layers.0.mlp."


def projection(x, weight, scales, routes, top_k, multiply_routes):
    count = routes.shape[1]
    dimension = weight.shape[1]
    ids = torch.arange(count, device=x.device, dtype=torch.int32).reshape(1, count)
    sorted_ids = torch.full((count, 16), count, device=x.device, dtype=torch.int32)
    sorted_ids[:, 0] = ids.flatten()
    experts = ids.flatten().contiguous()
    padded = torch.tensor([count * 16], device=x.device, dtype=torch.int32)
    output = torch.empty((1, count, dimension), device=x.device, dtype=torch.bfloat16)
    invoke_fused_moe_kernel(x, weight, None, output, None, scales, None, routes, ids,
                           sorted_ids.flatten().contiguous(), experts, padded, multiply_routes, top_k,
                           {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
                           tl.bfloat16, True, False, False, False, False, [128, 128])
    return output


def verify(model, device=1):
    torch.set_num_threads(4)
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model / "config.json").read_text())["text_config"]

    def weight(suffix):
        name = PREFIX + suffix
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    router = weight("gate.weight").cuda()
    sg, su, sd, sr = [weight(name + ".weight").cuda() for name in
                      ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate")]
    inputs, expected, routes = [], [], []
    vector = torch.arange(config["hidden_size"], device="cuda", dtype=torch.float32)
    for step in range(3):
        x = (.4 * torch.sin(vector * .013 + step * .27)).bfloat16().reshape(1, -1)
        inputs.append(x.float().cpu().flatten().tolist())
        probabilities, ids = moe_fused_gate(F.linear(x, router).float(), None, config["num_experts_per_tok"],
                                            scoring_func="softmax", renormalize=True)
        selected = ids.flatten().cpu().tolist()
        routes.append((selected, probabilities.flatten().cpu()))
        w13, s13, w2, s2 = [], [], [], []
        for expert in selected:
            name = f"experts.{expert}."
            gate, up = weight(name + "gate_proj.weight"), weight(name + "up_proj.weight")
            w13.append(torch.cat((gate.view(torch.uint8), up.view(torch.uint8))).view(torch.float8_e4m3fn))
            s13.append(torch.cat((weight(name + "gate_proj.weight_scale_inv"), weight(name + "up_proj.weight_scale_inv"))).float())
            w2.append(weight(name + "down_proj.weight")); s2.append(weight(name + "down_proj.weight_scale_inv").float())
        fc1 = projection(x, torch.stack(w13).cuda(), torch.stack(s13).cuda(), probabilities, len(selected), False)
        gate, up = fc1.reshape(len(selected), -1).chunk(2, dim=-1)
        mid = (F.silu(gate.float()) * up.float()).bfloat16()
        fc2 = projection(mid, torch.stack(w2).cuda(), torch.stack(s2).cuda(), probabilities, 1, True)
        routed = fc2.float().sum(dim=1).bfloat16()
        shared = F.linear((F.silu(F.linear(x, sg).float()) * F.linear(x, su).float()).bfloat16(), sd)
        shared = shared * torch.sigmoid(F.linear(x, sr))
        expected.append((routed + shared).float().cpu().flatten())
    with tempfile.TemporaryDirectory(prefix="jitllm-fp8-moe-") as directory:
        request = Path(directory) / "request.json"
        request.write_text(json.dumps({"inputs": inputs, "device": device, "gpu_cache_bytes": 128 << 20}))
        process = subprocess.run([sys.executable, str(ROOT / "jitllm_runtime.py"), "fp8-moe-probe", str(model), str(request)],
                                 capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr)
        actual = json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{')))
    comparisons = []
    for step, (ref, result) in enumerate(zip(expected, actual["outputs"], strict=True)):
        assert sorted(result["experts"]) == sorted(routes[step][0])
        weights = dict(zip(routes[step][0], routes[step][1].tolist(), strict=True))
        torch.testing.assert_close(torch.tensor(result["weights"]), torch.tensor([weights[i] for i in result["experts"]]), rtol=1e-5, atol=1e-7)
        output = torch.tensor(result["output"]); difference = output - ref
        comparison = {"step": step, "max_absolute_error": difference.abs().max().item(),
                      "rmse": difference.square().mean().sqrt().item(), "exact_fraction": (difference == 0).float().mean().item()}
        comparisons.append(comparison)
        print(comparison, flush=True)
        torch.testing.assert_close(output, ref, rtol=.008, atol=.0001)
    assert actual["forwards"] == 6 and actual["layer_builds"] == 1 and actual["evictions"] > 0
    assert actual["peak_bytes"] <= actual["budget_bytes"] and actual["intermediate_download_bytes"] == 0
    assert actual["resident_bytes_after_close"] == actual["native_contexts_after_close"] == actual["native_plans_after_close"] == actual["expert_registrations_after_close"] == 0
    return {"passed": True, "checkpoint": str(model), "reference": "Staged Triton block-FP8 W8A8 routed experts and BF16 shared expert",
            "routing_weight_before_bf16_fc2_store": True, "rtol": .008, "atol": .0001, "comparisons": comparisons, "execution": actual}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path); parser.add_argument("--device", type=int, default=1); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = json.dumps(verify(args.model, args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(report + "\n")
    print(report)

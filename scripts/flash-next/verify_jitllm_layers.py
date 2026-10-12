"""Compare native stateful PLE and hyper-connections with real BF16 checkpoint weights."""
import argparse
import bisect
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.srt.layers.hyperconnection import GatedResidual,HyperConnectionConfig

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "model.language_model.layers.1"


def verify(model,gpu_hc_ple=False,gpu_projections=False,device=1,hc_rows=1,fused_ple_decode=False,report_only=False,trace_hyper=False):
    if gpu_projections and not gpu_hc_ple:raise ValueError("--gpu-projections requires --gpu-hc-ple")
    torch.set_num_threads(4)
    torch.cuda.set_device(next((i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)),0))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}

    def weight(name):
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    def row(name, index, width):
        path = model / mapping[name]
        if path not in headers:
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                headers[path] = (8 + length, json.loads(stream.read(length)))
        base, header = headers[path]
        with path.open("rb") as stream:
            stream.seek(base + header[name]["data_offsets"][0] + index * width)
            raw = bytearray(stream.read(width))
        return torch.frombuffer(raw, dtype=torch.float8_e4m3fn).to(torch.bfloat16)

    config = json.loads((model / "config.json").read_text())["text_config"]
    hidden_size, branches, eps = config["hidden_size"], config["hc_count"], config["rms_norm_eps"]
    hyper_size = hidden_size * branches
    hidden = torch.sin(torch.arange(hyper_size, dtype=torch.float32) * .013).to(torch.bfloat16)
    block = torch.cos(torch.arange(hidden_size, dtype=torch.float32) * .021).to(torch.bfloat16)
    tokens = [248044, 9419, 1814, 0, 248044, 109266, 96748, 16, 17, 18, 19, 20]

    def norm(values, weights):
        grouped = values.float().reshape(branches, hidden_size)
        normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        return (normalized.flatten() * (1 + weights.float())).to(torch.bfloat16)

    h = PREFIX + ".attn_hyper_connection."
    hc=GatedResidual(HyperConnectionConfig(hc_count=branches,hidden_size=hidden_size,params_dtype=torch.bfloat16,
                                         hc_lowrank=config["hc_lowrank"],rms_norm_eps=eps,hc_per_branch_norm=True))
    hc=hc.to(device="cuda",dtype=torch.bfloat16)
    hc.load_state_dict({name:weight(h+name).cuda() for name in ["hc_norm.weight","input_mix_weight_down.weight","input_mix_weight_up.weight","block_inject_weight.weight"]})
    with torch.no_grad():
        mixed,residual=hc.mix(hidden.unsqueeze(0).repeat(hc_rows,1).cuda())
        combined=hc.combine(block.unsqueeze(0).repeat(hc_rows,1).cuda(),residual)[0].flatten().cpu()
        normed=residual[1][0].flatten().cpu();mixed=mixed[0].flatten().cpu()

    ple = PREFIX + ".ple."
    embeddings = ple + "ple_embedding."
    table = embeddings + "ngram_embedding."
    multipliers = weight(embeddings + "layer_multipliers")
    sizes = weight(embeddings + "ngram_heads_vocab_sizes")
    offsets = weight(embeddings + "ngram_heads_offsets")
    scale = weight(table + "weight_scale")
    shards = sorted((name for name in mapping if name.startswith(table + "shard_") and name.endswith(".weight")),
                    key=lambda name: int(name.split("shard_")[1].split(".")[0]))
    ends, total = [], 0
    for name in shards:
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            total += file.get_slice(name).get_shape()[0]
        ends.append(total)
    key_weight, value_weight = weight(ple + "key_proj.weight"), weight(ple + "value_proj.weight")
    norm_key, norm_query, norm_conv = weight(ple + "norm_key.weight"), weight(ple + "norm_query.weight"), weight(ple + "norm_conv.weight")
    conv = weight(ple + "conv1d.weight")
    state_length = (config["ple_conv_kernel_size"] - 1) * config["ngram_size"]
    conv_state = torch.zeros(hyper_size, state_length, dtype=torch.bfloat16)
    history = [config["eos_token_id"]] * config["ngram_size"]
    expected_ple = []
    for token in tokens:
        history = history[1:] + [token]
        context = torch.tensor(history, dtype=torch.int64)
        crossed_eos = False
        hashed = context[-1] * multipliers[0]
        ids = []
        for position in (1, 2):
            previous = int(context[-1-position])
            crossed_eos |= previous == config["eos_token_id"]
            hashed ^= torch.tensor(config["eos_token_id"] if crossed_eos else previous, dtype=torch.int64) * multipliers[position]
            start = (position-1)*config["heads_per_ngram"]
            ids += (hashed.remainder(sizes[start:start+8]) + offsets[start:start+8]).tolist()
        gathered = []
        for index in ids:
            shard = bisect.bisect_right(ends, index)
            begin = 0 if shard == 0 else ends[shard-1]
            gathered.append(row(shards[shard], index-begin, 160) * scale)
        embedding = torch.cat(gathered)
        keys = norm(F.linear(embedding, key_weight), norm_key).reshape(branches, hidden_size)
        query = norm(hidden, norm_query).reshape(branches, hidden_size)
        values = F.linear(embedding, value_weight)
        gate = (keys*query).sum(dim=-1, keepdim=True) / hidden_size**.5
        if fused_ple_decode:
            from sglang.kernels.ops.qwen4_ple import fused_qwen4_gate_value
            gated=fused_qwen4_gate_value(gate.unsqueeze(0).cuda().contiguous(),values.unsqueeze(0).cuda().contiguous()).flatten().cpu()
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt()*gate.sign()
            gated = (torch.sigmoid(gate)*values.unsqueeze(0)).flatten()
        normalized = norm(gated, norm_conv)
        conv_input = torch.cat([conv_state, normalized.unsqueeze(-1)], dim=-1)
        conv_output = F.conv1d(conv_input.unsqueeze(0), conv, dilation=config["ngram_size"], groups=hyper_size).flatten()
        expected_ple.append(gated + F.silu(conv_output))
        conv_state = conv_input[:, 1:]

    with tempfile.TemporaryDirectory(prefix="jitllm-layer-parity-") as temp:
        request = Path(temp) / "request.json"
        request.write_text(json.dumps({"hidden": hidden.float().tolist(), "block": block.float().tolist(), "tokens": tokens,
                                       "gpu_projections":gpu_projections,"device":device,"hc_rows":hc_rows,"fused_ple_decode":fused_ple_decode,"trace_hyper":trace_hyper}))
        command=[sys.executable,str(Path(__file__).resolve().parent/"jitllm_runtime.py"),"hc-ple-probe",str(model),str(request)] if gpu_hc_ple else [
                "java","-Xmx1g","-cp",str(ROOT/"target/classes"),"org.beehive.jitllm.integration.cli.FlashNextLayerProbe",str(model),str(request)]
        result = subprocess.run(command,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        lines=[line for line in result.stdout.splitlines() if line.startswith('{"')]
        if len(lines)!=1:raise RuntimeError("Expected one HC/PLE report: "+result.stdout)
        actual=json.loads(lines[0])

    comparisons = []
    for name, expected, observed in [("hyper_normalize", normed, actual["normalized"]), ("hyper_mix", mixed, actual["mixed"]), ("hyper_combine", combined, actual["combined"])]+[
            (f"ple_token_{i}", value, actual["ple"][i]) for i, value in enumerate(expected_ple)]:
        reference = expected.float()
        decoded = torch.tensor(observed)
        differences = decoded - reference
        close=torch.isclose(decoded,reference,rtol=.008,atol=.0001)
        comparisons.append({"operation": name, "max_absolute_error": differences.abs().max().item(), "rmse": differences.square().mean().sqrt().item(),
                            "exact_fraction": (differences == 0).float().mean().item(),"passed":bool(close.all()),"failed_values":int((~close).sum()),
                            "failures":[{"index":int(i),"actual":float(decoded.flatten()[i]),"reference":float(reference.flatten()[i])} for i in (~close).flatten().nonzero().flatten()[:10]]})
    for name, expected, observed in [("hyper_normalize", normed, actual["normalized"]), ("hyper_mix", mixed, actual["mixed"]), ("hyper_combine", combined, actual["combined"])]+[
            (f"ple_token_{i}", value, actual["ple"][i]) for i, value in enumerate(expected_ple)]:
        # BF16 inputs/outputs with different FP32 reduction orders can cross a rounding
        # boundary. Cancellation in gated + convolution output magnifies relative error;
        # retain a small absolute bound as well as a one-step relative BF16 bound.
        if not report_only:torch.testing.assert_close(torch.tensor(observed), expected.float(), rtol=.008, atol=.0001, msg=name)
    assert actual["cache_resident_bytes"] <= 1048576
    report={"checkpoint": str(model.resolve()), "tokens": tokens, "dtype": "BF16", "rtol": .008, "atol": .0001,"passed":all(c["passed"] for c in comparisons),
            "hc_rows":hc_rows,"hc_mix_precision":"F32 projections" if hc_rows<=16 else "BF16 projection boundaries",
            "fused_ple_decode":fused_ple_decode,"rollback_exact": actual["rollback_exact"], "reset_exact": actual["reset_exact"],
            "cache_resident_bytes": actual["cache_resident_bytes"],"page_misses":actual["page_misses"],"file_bytes_read":actual["file_bytes_read"],"comparisons": comparisons}
    if gpu_hc_ple:
        stats=actual["gpu_hc_ple"]
        assert stats["kernel_ns"]>0 and stats["state_host_download_bytes"]==0 and stats["resident_bytes_after_close"]==0
        report["gpu_hc_ple"]=stats
        report["gpu_projections"]=gpu_projections
        if gpu_projections:
            report["bf16_gemm_executions"]=actual["bf16_gemm_executions"]
            report["bf16_gemm_cache_bytes"]=actual["bf16_gemm_cache_bytes"]
            assert actual["projection_bytes_after_close"]==0
            report["projection_bytes_after_close"]=0
        for key in ("foreign_checkpoint_refused","reset_checkpoint_refused","independent_sequences","checkpoint_pool_bounded","use_after_close_refused","residual_add_exact"):
            assert actual[key];report[key]=actual[key]
    if trace_hyper:
        with torch.no_grad():
            normalized=residual[1]
            down=F.linear(normalized,hc.input_mix_weight_down.weight)
            low=F.silu(down/branches)
            up=F.linear(low,hc.input_mix_weight_up.weight)
            reconstructed=(torch.sigmoid(up.float()).reshape(hc_rows,branches,hidden_size)*normalized.float().reshape(hc_rows,branches,hidden_size)).mean(1).to(torch.bfloat16)[0].cpu()
        trace=actual["hyper_trace"]
        detail=[]
        for name,expected in [("down_bf16",down[0].cpu()),("low",low[0].cpu()),("up_bf16",up[0].cpu()),("reconstructed_mix",reconstructed)]:
            actual_tensor=torch.tensor(trace["down_raw"]).to(torch.bfloat16) if name=="down_bf16" else torch.tensor(trace["up_raw"]).to(torch.bfloat16) if name=="up_bf16" else torch.tensor(trace["low"]) if name=="low" else torch.tensor(actual["mixed"])
            diff=actual_tensor.float()-expected.float();indices=(diff!=0).nonzero().flatten()
            detail.append({"operation":name,"max_error":float(diff.abs().max()),"different_values":int(indices.numel()),
                           "examples":[{"index":int(i),"actual":float(actual_tensor[i]),"reference":float(expected[i])} for i in indices[:10]]})
        report["hyper_trace_comparisons"]=detail
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-hc-ple",action="store_true")
    parser.add_argument("--gpu-projections",action="store_true")
    parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--hc-rows",type=int,default=1)
    parser.add_argument("--fused-ple-decode",action="store_true")
    parser.add_argument("--report-only",action="store_true",help="Write failed numeric comparisons as passed=false without stopping")
    parser.add_argument("--trace-hyper",action="store_true")
    args = parser.parse_args()
    report = json.dumps(verify(args.model,args.gpu_hc_ple,args.gpu_projections,args.device,args.hc_rows,args.fused_ple_decode,args.report_only,args.trace_hyper), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)

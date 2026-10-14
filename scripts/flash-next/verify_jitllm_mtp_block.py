"""Compare the complete native MTP block, frozen sparse drafts and rollback with staged kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sgl_kernel import gemma_rmsnorm
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from verify_jitllm_fp8_moe import projection

SCRIPT_DIR = Path(__file__).resolve().parent
# The staged HC mix reduces with atomic FP32 adds, so neither the reference nor
# JITLLM is bit-deterministic across replays: JITLLM replays are exact with the
# deterministic legacy mix, while the native mix compounds BF16 rounding noise
# through carried state (observed worst case 2.5% of tensor scale over the
# teacher rows). Stage gates allow twice that, scaled by the reference tensor's
# magnitude; greedy tokens, sparse selections and other discrete decisions are
# still compared exactly.
REPLAY_ATOL = .004
REPLAY_RTOL = .05
# Free-running drafts feed their own hyper state back, so cross-implementation
# rounding noise compounds chaotically over draft steps (observed worst case
# 14% of tensor scale). With the deterministic legacy mix the same comparison
# is bit-exact at every stage, so draft correctness is gated on exact greedy
# tokens and sparse selections plus this gross-divergence bound.
DRAFT_RTOL = .25


def within_replay_tolerance(out, ref, rtol=REPLAY_RTOL):
    bound = REPLAY_ATOL + rtol * ref.abs().max()
    return bool(((out - ref).abs() <= bound).all())


def verify(model, device=1, trace_output=None, reference_replay_only=False, decode_reference="staged", teacher_fixture=None):
    torch.set_num_threads(4)
    gpu_name = os.environ.get("JITLLM_TORCH_GPU", "6000")
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if gpu_name in torch.cuda.get_device_name(i)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model / "config.json").read_text())["text_config"]
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    valid_vocabulary = max([*tokenizer["model"]["vocab"].values(), *(t["id"] for t in tokenizer["added_tokens"])]) + 1
    dim, branches, eps = config["hidden_size"], config["hc_count"], config["rms_norm_eps"]
    heads, kvheads, width = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    index_heads, index_width, ratio = config["indexer_n_heads"], config["indexer_head_dim"], config["indexer_compress_ratio"]
    rotary = int(width * config["rope_parameters"]["partial_rotary_factor"]); half = rotary // 2

    def weight(name, cuda=True):
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            value = file.get_tensor(name)
        return value.cuda() if cuda else value

    def embedding(token):
        name = "model.language_model.embed_tokens.weight"
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            return file.get_slice(name)[token].cuda().reshape(1, dim)

    def hc(prefix, combine):
        module = GatedResidual(HyperConnectionConfig(hc_count=branches, hidden_size=dim, params_dtype=torch.bfloat16,
                                                     hc_lowrank=config["hc_lowrank"], rms_norm_eps=eps, hc_per_branch_norm=True), use_combine=combine).to(device="cuda", dtype=torch.bfloat16)
        names = ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"]
        if combine:
            names.append("block_inject_weight.weight")
        module.load_state_dict({name: weight(prefix + "." + name) for name in names})
        return module

    attn_hc = hc("mtp.layers.0.attn_hyper_connection", True)
    mlp_hc = hc("mtp.layers.0.mlp_hyper_connection", True)
    mixer = hc("mtp.hyper_connection_mixer", False)
    en, hn, ew, hw = [weight("mtp." + name) for name in ["pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "fc_embedding.weight", "fc_hidden.weight"]]
    a = "mtp.layers.0.self_attn."
    qw, kw, vw, ow, qn, kn, iw, iqn, ikn = [weight(a + name) for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                                                              "q_norm.weight", "k_norm.weight", "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"]]
    m = "mtp.layers.0.mlp."
    rw, sg, su, sd, sr = [weight(m + name + ".weight") for name in ["gate", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate"]]
    head = weight("lm_head.weight")
    frequency = 1 / config["rope_parameters"]["rope_theta"] ** (torch.arange(0, rotary, 2, device="cuda", dtype=torch.float32) / rotary)
    keys, vals, fresh_keys, fresh_vals, queries, selections, pending, compressed = [], [], [], [], [], [], [], []
    captured = None
    decode_workspace = None
    if decode_reference == "staged":
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        decode_workspace = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def norm(x, w):
        return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * (1 + w.float())

    def rope(x, position):
        angle = position * frequency; c, s = angle.cos(), angle.sin()
        left, right = x[..., :half], x[..., half:rotary]
        return torch.cat([left * c - right * s, right * c + left * s, x[..., rotary:]], dim=-1).bfloat16()

    def moe(x):
        routes, ids = moe_fused_gate(F.linear(x, rw).float(), None, config["num_experts_per_tok"], scoring_func="softmax", renormalize=True)
        selected = ids.flatten().cpu().tolist(); w13, s13, w2, s2 = [], [], [], []
        for expert in selected:
            p = m + f"experts.{expert}."
            gate, up = weight(p + "gate_proj.weight", False), weight(p + "up_proj.weight", False)
            w13.append(torch.cat([gate.view(torch.uint8), up.view(torch.uint8)]).view(torch.float8_e4m3fn))
            s13.append(torch.cat([weight(p + "gate_proj.weight_scale_inv", False), weight(p + "up_proj.weight_scale_inv", False)]))
            w2.append(weight(p + "down_proj.weight", False)); s2.append(weight(p + "down_proj.weight_scale_inv", False))
        fc1 = projection(x, torch.stack(w13).cuda(), torch.stack(s13).cuda(), routes, len(selected), False)
        gate, up = fc1.reshape(len(selected), -1).chunk(2, -1)
        mid = (F.silu(gate.float()) * up.float()).bfloat16()
        routed = projection(mid, torch.stack(w2).cuda(), torch.stack(s2).cuda(), routes, 1, True).float().sum(1).bfloat16()
        shared = F.linear((F.silu(F.linear(x, sg).float()) * F.linear(x, su).float()).bfloat16(), sd) * torch.sigmoid(F.linear(x, sr))
        return routed + shared

    def forward(token, target_hidden, position, draft):
        nonlocal captured
        fused = (F.linear(gemma_rmsnorm(embedding(token), en, eps), ew).unsqueeze(1)
                 + F.linear(gemma_rmsnorm(target_hidden.reshape(1, -1), hn, eps).reshape(1, branches, dim), hw)).reshape(1, -1)
        trace = {"fusion": fused}
        x, residual = attn_hc.mix(fused); trace["attention_mix"] = x
        qg = F.linear(x, qw).reshape(heads, 2 * width)
        q = rope(norm(qg[:, :width], qn), position); gate = qg[:, width:]
        k = rope(norm(F.linear(x, kw).reshape(kvheads, width), kn), position)
        v = F.linear(x, vw).reshape(kvheads, width)
        fresh_keys.append(k); fresh_vals.append(v); queries.append(q)
        keys.append(k.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()); vals.append(v.float().clamp(-448, 448).to(torch.float8_e4m3fn).float())
        if draft:
            selected = captured + list(range(len(tokens), position + 1))
        else:
            ip = F.linear(x, iw).flatten(); iq = rope(norm(ip[:index_heads * index_width].reshape(index_heads, index_width), iqn), position)
            pending.append(ip[index_heads * index_width:])
            if (position + 1) % ratio == 0:
                compressed.append(rope(norm(torch.stack(pending[-ratio:]).float().mean(0), ikn), position + 1 - ratio))
            if compressed:
                scores = torch.relu(iq.float() @ torch.stack(compressed).float().T).sum(0) / index_width ** .5
                blocks = torch.topk(scores, min(len(compressed), config["indexer_budget"] // ratio)).indices.tolist()
            else:
                blocks = []
            selected = [block * ratio + i for block in blocks for i in range(ratio)] + list(range((position + 1) // ratio * ratio, position + 1))
            captured = selected.copy()
        selections.append(selected)
        if draft:
            if decode_reference == "staged":
                # QSA's SM120 paged path dequantizes the FP8 pool into BF16,
                # pads every selected row to 64-token pages, and calls the staged FlashInfer decode API.
                stride = ((config["indexer_budget"] + ratio - 1 + 3 + 63) // 64) * 64
                packed_k = torch.zeros((stride, kvheads, width), device="cuda", dtype=torch.bfloat16)
                packed_v = torch.zeros_like(packed_k)
                packed_k[:len(selected)] = torch.stack(keys)[selected].bfloat16()
                packed_v[:len(selected)] = torch.stack(vals)[selected].bfloat16()
                kc = packed_k.view(-1, 64, kvheads, width).permute(0, 2, 1, 3)
                vc = packed_v.view(-1, 64, kvheads, width).permute(0, 2, 1, 3)
                attended = trtllm_batch_decode_with_kv_cache(
                    query=q.unsqueeze(0).contiguous(), kv_cache=(kc, vc), workspace_buffer=decode_workspace,
                    block_tables=torch.arange(stride // 64, device="cuda", dtype=torch.int32).reshape(1, -1),
                    seq_lens=torch.tensor([len(selected)], device="cuda", dtype=torch.int32),
                    max_seq_len=stride, bmm1_scale=width ** -.5, bmm2_scale=1.)[0]
            else:
                cached_k = torch.stack(keys)[selected].repeat_interleave(heads // kvheads, 1)
                cached_v = torch.stack(vals)[selected].repeat_interleave(heads // kvheads, 1)
                scores = torch.einsum("hd,nhd->hn", q.float(), cached_k.float()) / width ** .5
                attended = torch.einsum("hn,nhd->hd", torch.softmax(scores, -1), cached_v).bfloat16()
        else:
            topk = torch.full((position + 1, position + 1), -1, device="cuda", dtype=torch.int32)
            for row, selection in enumerate(selections):
                topk[row, :len(selection)] = torch.tensor(selection, device="cuda", dtype=torch.int32)
            attended = sparse_gqa_fwd_interface_triton(torch.stack(queries).contiguous(), torch.stack(fresh_keys).contiguous(), torch.stack(fresh_vals).contiguous(),
                                                      position + 1, topk, torch.tensor([0, position + 1], device="cuda", dtype=torch.int32), width ** -.5)[-1]
        gated = fused_sigmoid_mul(attended.flatten().unsqueeze(0), gate.unsqueeze(0), inplace=False).reshape(1, -1)
        attended = F.linear(gated, ow); trace["attention"] = attended
        hyper = attn_hc.combine(attended, residual); trace["after_attention"] = hyper
        x, residual = mlp_hc.mix(hyper); trace["mlp_mix"] = x
        routed = moe(x); trace["moe"] = routed
        hyper = mlp_hc.combine(routed, residual); trace["hyper_hidden"] = hyper
        hidden, _ = mixer.mix(hyper); trace["hidden"] = hidden
        logits = F.linear(hidden, head).float().flatten(); logits[valid_vocabulary:] = -torch.inf
        return trace, int(logits.argmax()), hyper

    if teacher_fixture:
        fixture = json.loads(teacher_fixture.read_text())
        tokens, draft_tokens = fixture["tokens"], fixture["draft_tokens"]
        if not tokens or len(draft_tokens) != 3 or len(fixture["target_hidden"]) != len(tokens):
            raise ValueError("Teacher fixture requires paired inputs and three draft tokens")
        if any(len(row) != dim * branches for row in fixture["target_hidden"]):
            raise ValueError("Teacher fixture has an invalid HC width")
        target_hidden = [torch.tensor(row, device="cuda", dtype=torch.bfloat16) for row in fixture["target_hidden"]]
    else:
        tokens = [17, 478, 220, 20, 248044]
        draft_tokens = [9419, 1814, 0]
        vector = torch.arange(dim, device="cuda", dtype=torch.float32)
        amplitudes = torch.arange(1, branches + 1, device="cuda", dtype=torch.float32).reshape(branches, 1) / 2
        target_hidden = [(amplitudes * torch.cos(vector * .021 + row * .11)).flatten().bfloat16() for row in range(len(tokens))]
    expected = []
    with torch.no_grad():
        for position, (token, hidden) in enumerate(zip(tokens, target_hidden, strict=True)):
            trace, greedy, last = forward(token, hidden, position, False); expected.append((trace, greedy))
        for step, token in enumerate(draft_tokens):
            trace, greedy, last = forward(token, last, len(tokens) + step, True); expected.append((trace, greedy))
    if reference_replay_only:
        for values in [keys, vals, fresh_keys, fresh_vals, queries, selections, pending, compressed]: values.clear()
        captured = None
        repeated=[]
        with torch.no_grad():
            for position,(token,hidden) in enumerate(zip(tokens,target_hidden,strict=True)):
                trace,greedy,last=forward(token,hidden,position,False);repeated.append((trace,greedy))
            for step,token in enumerate(draft_tokens):
                trace,greedy,last=forward(token,last,len(tokens)+step,True);repeated.append((trace,greedy))
        cases=[];failures=[]
        for step,((first,greedy),(second,replayed_greedy)) in enumerate(zip(expected,repeated,strict=True)):
            stages={}
            for name,a in first.items():
                a=a.float().cpu().flatten();b=second[name].float().cpu().flatten();diff=a-b
                stages[name]={"max_absolute_error":diff.abs().max().item(),"exact_fraction":(diff==0).float().mean().item()}
                if not within_replay_tolerance(b,a):failures.append({"step":step,"stage":name,"error":f"max_absolute_error {diff.abs().max().item()} exceeds replay bound {REPLAY_ATOL + REPLAY_RTOL * a.abs().max().item()}"})
            cases.append({"step":step,"stages":stages,"greedy":greedy,"replayed_greedy":replayed_greedy})
        return {"passed":not failures,"reference_replay":True,"failures":failures,"cases":cases}
    with tempfile.TemporaryDirectory(prefix="jitllm-mtp-block-") as directory:
        request = Path(directory) / "request.json"
        request.write_text(json.dumps({"tokens": tokens, "target_hidden": [h.float().cpu().tolist() for h in target_hidden], "draft_tokens": draft_tokens,
                                       "device": device, "with_logits": True, "native_qsa_decode": decode_reference == "staged"}))
        process = subprocess.run([sys.executable, str(SCRIPT_DIR / "jitllm_runtime.py"), "mtp-block-probe", str(model), str(request)], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr + "\n" + process.stdout)
        actual = json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{"')))
    debug = {"actual": actual, "expected": [{"greedy": greedy, "stages": {name: values.float().cpu().flatten().tolist() for name, values in trace.items()}} for trace, greedy in expected]}
    if trace_output:
        trace_output.parent.mkdir(parents=True, exist_ok=True); trace_output.write_text(json.dumps(debug))
    comparisons, failures = [], []
    for step, ((trace, greedy), observed) in enumerate(zip(expected, actual.pop("teacher") + actual.pop("drafts"), strict=True)):
        stages = {}
        for name, values in trace.items():
            ref = values.float().cpu().flatten(); out = torch.tensor(observed[name]); diff = out - ref
            stages[name] = {"max_absolute_error": diff.abs().max().item(), "exact_fraction": (diff == 0).float().mean().item()}
            rtol = DRAFT_RTOL if step >= len(tokens) else REPLAY_RTOL
            if not within_replay_tolerance(out, ref, rtol):
                failures.append({"step": step, "stage": name, "error": f"max_absolute_error {diff.abs().max().item()} exceeds replay bound {REPLAY_ATOL + rtol * ref.abs().max().item()}"})
        assert sorted(observed["selected_tokens"]) == sorted(selections[step])
        if step == len(tokens) - 1 or step >= len(tokens):
            if observed["greedy"] != greedy:
                failures.append({"step": step, "observed_greedy": observed["greedy"], "expected_greedy": greedy})
        comparisons.append({"step": step, "draft": step >= len(tokens), "greedy": greedy, "stages": stages})
        print(comparisons[-1], flush=True)
    assert actual["rollback_replay_close"] and actual["reset_replay_close"] and actual["failure_retry_close"]
    assert actual["resources_after_close"] == actual["kv_download_bytes"] == 0 and actual["native_calls"] > 0
    return {"passed": not failures, "failures": failures, "checkpoint": str(model), "reference": "Staged CUDA Gemma/HC/sparse-prefill/fused-gate/Triton-W8A8",
            "decode_reference": "FlashInfer paged BF16 decode over FP8-rounded KV (XQA on SM120)" if decode_reference == "staged" else "PyTorch FP32 decode over FP8-rounded KV",
            "teacher_tokens": tokens, "draft_tokens": draft_tokens, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL,
            "tolerance": "teacher per-stage |diff| <= atol + rtol*max|reference|; draft steps use draft_rtol (free-running hyper feedback compounds rounding noise; legacy-mix comparison is bit-exact); greedy and selections exact",
            "draft_rtol": DRAFT_RTOL, "hc_mix": "native" if actual.get("native_hc_mix_calls", 0) > 0 else "legacy",
            "comparisons": comparisons, "execution": actual,
            "teacher_input_source": "supplied target HC fixture" if teacher_fixture else "synthetic cosine HC vectors",
            "teacher_fixture_sha256": hashlib.sha256(teacher_fixture.read_bytes()).hexdigest() if teacher_fixture else None,
            "scope": "MTP arithmetic and state; target verification and API generation are checked separately"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("model", type=Path); parser.add_argument("--device", type=int, default=1); parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--reference-replay-only", action="store_true")
    parser.add_argument("--decode-reference", choices=["staged", "pytorch"], default="staged")
    parser.add_argument("--teacher-fixture", type=Path, help="Real teacher inputs captured by speculative-probe")
    args = parser.parse_args(); result = verify(args.model, args.device, args.trace_output, args.reference_replay_only, args.decode_reference, args.teacher_fixture); report = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(report + "\n")
    print(report)
    if not result["passed"]:
        raise SystemExit(1)

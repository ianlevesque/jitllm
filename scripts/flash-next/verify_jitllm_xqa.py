"""Compare the Java/TornadoVM XQA binding with staged FlashInfer over sparse/page boundaries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

SCRIPT_DIR = Path(__file__).resolve().parent


def verify(device=1):
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    capacity, heads, kv_heads, dimension = 2112, 24, 2, 256
    counts = [1, 6, 63, 64, 65, 2048, 2051, 2054]
    query = torch.randn((1, heads, dimension), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((capacity, kv_heads, dimension), device="cuda").to(torch.float8_e4m3fn).bfloat16()
    values = torch.randn_like(keys.float()).to(torch.float8_e4m3fn).bfloat16()
    table = torch.arange(capacity // 64, device="cuda", dtype=torch.int32).reshape(1, -1)
    workspace = torch.zeros(16 << 20, device="cuda", dtype=torch.uint8)
    expected = {}
    for count in counts:
        expected[count] = trtllm_batch_decode_with_kv_cache(
            query=query, kv_cache=(keys.view(-1, 64, kv_heads, dimension).permute(0, 2, 1, 3), values.view(-1, 64, kv_heads, dimension).permute(0, 2, 1, 3)),
            workspace_buffer=workspace, block_tables=table, seq_lens=torch.tensor([count], device="cuda", dtype=torch.int32),
            max_seq_len=capacity, bmm1_scale=dimension ** -.5, bmm2_scale=1., enable_pdl=False).float().cpu().flatten()
    with tempfile.TemporaryDirectory(prefix="jitllm-xqa-") as temporary:
        directory = Path(temporary)
        request = {"device": device, "counts": counts}
        for name, tensor in [("query", query), ("keys", keys), ("values", values)]:
            path = directory / (name + ".bf16")
            path.write_bytes(tensor.cpu().contiguous().view(torch.int16).numpy().tobytes())
            request[name] = str(path)
        path = directory / "request.json"; path.write_text(json.dumps(request))
        process = subprocess.run([sys.executable, str(SCRIPT_DIR / "jitllm_runtime.py"), "xqa-probe", str(path)], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr + "\n" + process.stdout)
        actual = json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{"')))
    for case in actual["cases"]:
        observed = torch.tensor(case.pop("output")); reference = expected[case["selected_tokens"]]
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        case["exact"] = True
        assert case["replay_exact"] and case["caller_thread_exact"]
    assert actual["resources_after_close"] == 0 and actual["native_calls"] == 6 * len(counts) and actual["kernel_ns"] > 0
    actual["reference"] = "Staged FlashInfer XQA BF16 decode over FP8-rounded KV"
    actual["scope"] = "Native Java ABI, sparse/page boundaries, graph replay, caller-thread stream changes, library reload and resource release. Not full-model speculative acceptance."
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", type=int, default=1); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = json.dumps(verify(args.device), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(report + "\n")
    print(report)

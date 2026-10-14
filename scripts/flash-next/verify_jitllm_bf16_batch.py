"""Real-weight native BF16 GEMM with distinct rows, graph replay and caller-thread isolation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent


def verify(model: Path) -> dict:
    torch.set_num_threads(4)
    torch.cuda.set_device(next((i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)),0))
    mapping=json.loads((model/"model.safetensors.index.json").read_text())["weight_map"]
    reports=[]
    for projection in ("input_mix_weight_down","input_mix_weight_up"):
        prefix="model.language_model.layers.1.attn_hyper_connection."+projection
        with safe_open(model/mapping[prefix+".weight"],framework="pt",device="cpu") as file:
            weight=file.get_tensor(prefix+".weight").cuda()
        for batch in (1,33,64):
            cols=torch.arange(weight.shape[1],dtype=torch.float32)[None,:]
            rows=torch.arange(batch,dtype=torch.float32)[:,None]
            values=(torch.sin(cols*.017+rows*.19)+.1*torch.cos(cols*.003-rows*.11)).to(torch.bfloat16)
            expected=torch.nn.functional.linear(values.cuda(),weight).float().cpu()
            with tempfile.TemporaryDirectory(prefix="jitllm-bf16-batch-") as temp:
                request=Path(temp)/"request.json"
                request.write_text(json.dumps({"matrix":prefix,"batch_rows":batch,"input":values.float().flatten().tolist(),"device":1}))
                result=subprocess.run([sys.executable,str(ROOT/"jitllm_runtime.py"),"bf16-batch-probe",str(model),str(request)],capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stderr+result.stdout)
                lines=[line for line in result.stdout.splitlines() if line.startswith('{"')]
                if len(lines)!=1:raise RuntimeError("Expected one BF16 batch report: "+result.stdout)
                actual=json.loads(lines[0])
            observed=torch.tensor(actual.pop("output")).reshape(expected.shape)
            difference=observed-expected
            torch.testing.assert_close(observed,expected,rtol=.008,atol=.0001)
            for flag in ("cache_replay_exact","caller_thread_exact","source_bytes_unchanged"):assert actual[flag]
            assert actual["resident_bytes_after_close"]==0 and actual["kernel_ns"]>0 and actual["executions"]==4
            actual.update(max_absolute_error=float(difference.abs().max()),rmse=float(difference.square().mean().sqrt()),exact_fraction=float((difference==0).float().mean()),passed=True)
            reports.append(actual)
    return {"passed":True,"checkpoint":str(model.resolve()),"dtype":"BF16","cuBLAS_workspace_bytes":33554432,"rtol":.008,"atol":.0001,
            "scope":"Distinct-row hardware BF16 GEMM on real layer-1 HC matrices, graph reuse, caller-thread ownership and checkpoint-byte preservation. Not full-model prefill qualification.","reports":reports}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args();report=json.dumps(verify(args.model),indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(report+"\n")
    print(report)

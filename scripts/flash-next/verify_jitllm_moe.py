"""Compare native W4A4 MoE execution with the staged FlashInfer CUTLASS fused kernel."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys

import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe
from safetensors import safe_open
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

ROOT=Path(__file__).resolve().parents[2]
PREFIX="model.language_model.layers.0.mlp."


def verify(model,gpu=False,native=False,device=1):
    torch.set_num_threads(4);torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    mapping=json.loads((model/"model.safetensors.index.json").read_text())["weight_map"]
    config=json.loads((model/"config.json").read_text())["text_config"]
    def weight(name):
        name=PREFIX+name
        with safe_open(model/mapping[name],framework="pt",device="cpu") as file:return file.get_tensor(name)
    def swizzle(scale):
        batches,rows,cols=scale.shape
        assert rows%128==0 and cols%4==0
        return scale.view(torch.uint8).reshape(batches,rows//128,4,32,cols//4,4).permute(0,1,4,3,2,5).contiguous().reshape(batches,rows,cols).view(torch.int32)
    input_scale=0.;mid_scale=0.
    for expert in range(config["num_experts"]):
        input_scale=max(input_scale,weight(f"experts.{expert}.gate_proj.input_scale").item(),weight(f"experts.{expert}.up_proj.input_scale").item())
        mid_scale=max(mid_scale,weight(f"experts.{expert}.down_proj.input_scale").item())
    vector=torch.arange(config["hidden_size"],dtype=torch.float32)
    inputs=[(.4*torch.sin(vector*.013+step*.27)).to(torch.bfloat16) for step in range(3)]
    router=weight("gate.weight")
    expected=[];routes=[];probabilities=[]
    shared_gate,shared_up,shared_down,shared_router=weight("shared_expert.gate_proj.weight"),weight("shared_expert.up_proj.weight"),weight("shared_expert.down_proj.weight"),weight("shared_expert_gate.weight")
    for x in inputs:
        logits=F.linear(x,router).float()
        weights,ids=moe_fused_gate(logits.unsqueeze(0).cuda(),None,config["num_experts_per_tok"],scoring_func="softmax",renormalize=True)
        weights,ids=weights.flatten().cpu(),ids.flatten().cpu()
        routes.append(ids.tolist());probabilities.append(weights)
        w13=[];sf13=[];alpha1=[];w2=[];sf2=[];alpha2=[]
        for expert in ids.tolist():
            name=f"experts.{expert}."
            gate_scale=weight(name+"gate_proj.weight_scale_2").item();up_scale=weight(name+"up_proj.weight_scale_2").item()
            assert gate_scale==up_scale,"Staged single-alpha CUTLASS path requires matching gate/up scales"
            # The staged CUTLASS loader stores [Up, Gate]. Only selected experts are
            # remapped into this test tensor; original router weights/scales are retained.
            w13.append(torch.cat([weight(name+"up_proj.weight"),weight(name+"gate_proj.weight")],dim=0))
            sf13.append(torch.cat([weight(name+"up_proj.weight_scale"),weight(name+"gate_proj.weight_scale")],dim=0))
            alpha1.append(gate_scale*input_scale)
            w2.append(weight(name+"down_proj.weight"));sf2.append(weight(name+"down_proj.weight_scale"))
            alpha2.append(weight(name+"down_proj.weight_scale_2").item()*mid_scale)
        quant_scales=[torch.tensor(1/input_scale,device="cuda"),swizzle(torch.stack(sf13).cuda()),torch.tensor(alpha1,device="cuda"),
                      torch.tensor(1/mid_scale,device="cuda"),swizzle(torch.stack(sf2).cuda()),torch.tensor(alpha2,device="cuda")]
        routed=cutlass_fused_moe(input=x.unsqueeze(0).cuda(),token_selected_experts=torch.arange(len(ids),device="cuda",dtype=torch.int32).unsqueeze(0),
                                  token_final_scales=weights.unsqueeze(0).cuda(),fc1_expert_weights=torch.stack(w13).cuda().view(torch.int64),
                                  fc2_expert_weights=torch.stack(w2).cuda().view(torch.int64),output_dtype=torch.bfloat16,quant_scales=quant_scales,tune_max_num_tokens=1,
                                  use_fused_finalize=False)[0].flatten().cpu()
        gate=F.linear(x,shared_gate);up=F.linear(x,shared_up)
        activated=(F.silu(gate.float())*up.float()).to(torch.bfloat16)
        shared=F.linear(activated,shared_down)*torch.sigmoid(F.linear(x,shared_router))
        expected.append((routed+shared).float())
    with tempfile.TemporaryDirectory(prefix="jitllm-moe-") as temp:
        request=Path(temp)/"request.json"
        request.write_text(json.dumps({"mode":"moe","inputs":[x.float().tolist() for x in inputs],"gpu_nvfp4":native,"device":device}))
        command=[sys.executable,str(Path(__file__).resolve().parent/"jitllm_runtime.py"),"moe-probe",str(model),str(request)] if gpu else ["java","-Xmx1g","-cp",str(ROOT/"target/classes"),"org.beehive.jitllm.integration.cli.FlashNextLayerProbe",str(model),str(request)]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        actual=json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"')))
    assert actual["input_scale"]==input_scale and actual["intermediate_scale"]==mid_scale
    comparisons=[];failures=[]
    for step,(reference,result) in enumerate(zip(expected,actual["outputs"],strict=True)):
        assert sorted(result["experts"])==sorted(routes[step]),(step,result["experts"],routes[step])
        expected_weights={expert:float(value) for expert,value in zip(routes[step],probabilities[step],strict=True)}
        torch.testing.assert_close(torch.tensor(result["weights"]),torch.tensor([expected_weights[expert] for expert in result["experts"]]),rtol=1e-5,atol=1e-7)
        observed=torch.tensor(result["output"]);difference=observed-reference
        try:torch.testing.assert_close(observed,reference,rtol=.008,atol=.0001)
        except AssertionError as mismatch:failures.append({"step":step,"error":str(mismatch)})
        comparisons.append({"step":step,"experts":routes[step],"max_absolute_error":difference.abs().max().item(),"rmse":difference.square().mean().sqrt().item(),
                            "relative_rmse":difference.square().mean().sqrt().item()/reference.square().mean().sqrt().item(),"exact_fraction":(difference==0).float().mean().item()})
    return {"checkpoint":str(model.resolve()),"reference":"FlashInfer CUTLASS NVFP4 fused MoE + BF16 shared expert","device":torch.cuda.get_device_name(),"gpu_projections":actual.get("gpu_projections"),"native_nvfp4":native,
            "input_scale":input_scale,"intermediate_scale":mid_scale,"activation_quantization":"NVFP4 W4A4","fused_finalize":False,"rtol":.008,"atol":.0001,
            "cache_resident_bytes":actual["cache_resident_bytes"],"passed":not failures,"failures":failures,"comparisons":comparisons}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--gpu-projections",action="store_true");parser.add_argument("--native-nvfp4",action="store_true");parser.add_argument("--device",type=int,default=1)
    args=parser.parse_args()
    if args.native_nvfp4 and not args.gpu_projections:parser.error("--native-nvfp4 requires --gpu-projections")
    result=verify(args.model,args.gpu_projections,args.native_nvfp4,args.device);report=json.dumps(result,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(report+"\n")
    print(report)
    if not result["passed"]:raise SystemExit(1)

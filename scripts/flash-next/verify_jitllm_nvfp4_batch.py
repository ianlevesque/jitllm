"""Check native cuBLASLt packed NVFP4 GEMM against FlashInfer CUTLASS with real weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.gemm import mm_fp4

ROOT=Path(__file__).resolve().parent


def verify(model,matrices,batches,device=1):
    torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)))
    torch.set_num_threads(4)
    mapping=json.loads((model/"model.safetensors.index.json").read_text())["weight_map"]
    def weight(name):
        with safe_open(model/mapping[name],framework="pt",device="cpu") as file:return file.get_tensor(name)
    def swizzle(scale):
        rows,columns=scale.shape
        assert rows%128==0 and columns%4==0
        return scale.view(torch.uint8).reshape(rows//128,4,32,columns//4,4).permute(0,3,2,1,4).contiguous().reshape(rows,columns)
    cases=[]
    for prefix in matrices:
        w=weight(prefix+".weight");sf=swizzle(weight(prefix+".weight_scale"));alpha=weight(prefix+".weight_scale_2").item()
        scale=weight(prefix+".input_scale").item();columns=w.shape[1]*2
        vector=torch.arange(columns,dtype=torch.float32)
        for batch in batches:
            values=torch.stack([(.4*torch.sin(vector*.013+row*.17)).to(torch.bfloat16) for row in range(batch)])
            packed,scales=fp4_quantize(values.cuda(),torch.tensor(1/scale,device="cuda"))
            expected=mm_fp4(packed,w.cuda().T,scales,sf.cuda(),torch.tensor(alpha*scale,device="cuda"),out_dtype=torch.bfloat16,backend="cutlass").float().cpu().flatten()
            with tempfile.TemporaryDirectory(prefix="jitllm-nvfp4-batch-") as temp:
                request=Path(temp)/"request.json"
                request.write_text(json.dumps({"matrix":prefix,"batch_rows":batch,"device":device,"input_scale":scale,"input":values.float().flatten().tolist()}))
                result=subprocess.run([sys.executable,str(ROOT/"jitllm_runtime.py"),"nvfp4-batch-probe",str(model),str(request)],capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stderr+"\n"+result.stdout)
                lines=[line for line in result.stdout.splitlines() if line.startswith('{"')]
                if len(lines)!=1:raise RuntimeError("Expected one native NVFP4 report: "+result.stdout)
                actual=json.loads(lines[0])
            raw_scales=scales.view(torch.uint8).cpu().reshape(-1,columns//16)
            raw_scales=raw_scales.reshape(-1,columns//64,32,4,4).permute(0,3,2,1,4).contiguous().reshape(-1,columns//16)[:batch]
            packed_observed=torch.frombuffer(bytearray.fromhex(actual.pop("packed_values_hex")),dtype=torch.uint8).reshape(batch,-1)
            packed_expected=packed.view(torch.uint8).cpu()
            if not torch.equal(packed_observed,packed_expected):
                diff=(packed_observed!=packed_expected).nonzero();coords=diff.tolist()
                raise AssertionError(("packed activations differ",prefix,batch,coords,packed_observed[diff[:,0],diff[:,1]].tolist(),packed_expected[diff[:,0],diff[:,1]].tolist()))
            assert bytes.fromhex(actual.pop("packed_scales_hex"))==raw_scales.numpy().tobytes(),("activation block scales differ",prefix,batch)
            observed=torch.tensor(actual.pop("output"));difference=observed-expected
            torch.testing.assert_close(observed,expected,rtol=.008,atol=.0001)
            assert actual["native_calls"]>=4 and actual["resident_bytes_after_close"]==0 and actual["native_contexts_after_close"]==0 and actual["native_plans_after_close"]==0
            assert actual["cache_replay_exact"] and actual["caller_thread_exact"] and actual["source_bytes_unchanged"]
            actual.update({"max_absolute_error":difference.abs().max().item(),"exact_fraction":(difference==0).float().mean().item(),"input_scale":scale,"passed":True})
            cases.append(actual)
            print(json.dumps({"matrix":prefix,"batch":batch,"max_error":actual["max_absolute_error"]}),file=sys.stderr,flush=True)
    return {"checkpoint":str(model.resolve()),"reference":"FlashInfer CUTLASS packed NVFP4 GEMM","execution":"CUDA 13 cuBLASLt E2M1/VEC16_UE4M3, BF16 output","rtol":.008,"atol":.0001,"passed":True,"cases":cases}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path);parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--matrices",nargs="+",default=[f"model.language_model.layers.0.mlp.experts.0.{name}_proj" for name in ("gate","up","down")]);parser.add_argument("--batches",nargs="+",type=int,default=[1,3,33,128])
    args=parser.parse_args();report=verify(args.model,args.matrices,args.batches,args.device);encoded=json.dumps(report,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded+"\n")
    print(encoded)

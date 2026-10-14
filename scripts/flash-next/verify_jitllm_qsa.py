"""Compare native QSA projection, compression, sparse attention and MRoPE with real weights."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
PREFIX="model.language_model.layers.3.self_attn."


def verify(model,fp8=False,fused=False,fresh=False,gpu=False,projected=False,device=1):
    torch.set_num_threads(4)
    if fused:
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
        torch.cuda.set_device(next((i for i in range(torch.cuda.device_count()) if "5090" in torch.cuda.get_device_name(i)),0))
    if fresh:
        if not fused:raise ValueError("Fresh prefill verification requires --fused-precision")
        from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton
    mapping=json.loads((model/"model.safetensors.index.json").read_text())["weight_map"]
    config=json.loads((model/"config.json").read_text())["text_config"]
    def weight(name):
        name=PREFIX+name
        with safe_open(model/mapping[name],framework="pt",device="cpu") as file:return file.get_tensor(name)
    dim,heads,kv_heads=config["head_dim"],config["num_attention_heads"],config["num_key_value_heads"]
    index_dim,index_heads,ratio=config["indexer_head_dim"],config["indexer_n_heads"],config["indexer_compress_ratio"]
    rotary=int(dim*config["partial_rotary_factor"]);half=rotary//2
    q_weight,k_weight,v_weight,o_weight=weight("q_proj.weight"),weight("k_proj.weight"),weight("v_proj.weight"),weight("o_proj.weight")
    q_norm,k_norm=weight("q_norm.weight"),weight("k_norm.weight")
    i_weight,iq_norm,ik_norm=weight("indexer.index_qk_proj.weight"),weight("indexer.q_layernorm.weight"),weight("indexer.k_layernorm.weight")
    indices=torch.arange(config["hidden_size"],dtype=torch.float32)
    inputs=[torch.sin(indices*.011+step*.15).to(torch.bfloat16) for step in range(12)]
    positions=[[step,step//2,step%7] for step in range(len(inputs))]
    axis=torch.zeros(half,dtype=torch.long)
    section=config["rope_parameters"]["mrope_section"]
    pair=torch.arange(half)
    axis[(pair%3==1)&(pair<section[1]*3)]=1;axis[(pair%3==2)&(pair<section[2]*3)]=2
    frequency=1/(config["rope_parameters"]["rope_theta"]**(torch.arange(0,rotary,2,dtype=torch.float32)/rotary))
    def norm(x,w):
        output=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+config["rms_norm_eps"])*(1+w.float())
        return output if fused else output.to(torch.bfloat16)
    def rope(x,coordinates):
        angles=torch.tensor(coordinates,dtype=torch.float32)[axis]*frequency
        cosine,sine=angles.cos(),angles.sin()
        if not fused:cosine,sine=cosine.to(torch.bfloat16),sine.to(torch.bfloat16)
        left,right=x[...,:half],x[...,half:rotary]
        return torch.cat([left*cosine-right*sine,right*cosine+left*sine,x[...,rotary:]],dim=-1).to(torch.bfloat16)
    key_cache,value_cache,pending,compressed=[],[],[],[]
    fresh_keys,fresh_values,queries=[],[],[]
    expected,selections=[],[]
    for step,(x,coordinates) in enumerate(zip(inputs,positions,strict=True)):
        q_gate=F.linear(x,q_weight).reshape(heads,2*dim)
        q=rope(norm(q_gate[:,:dim],q_norm),coordinates)
        gate=q_gate[:,dim:]
        k=rope(norm(F.linear(x,k_weight).reshape(kv_heads,dim),k_norm),coordinates)
        v=F.linear(x,v_weight).reshape(kv_heads,dim)
        fresh_keys.append(k);fresh_values.append(v);queries.append(q)
        if fp8:
            k=k.float().clamp(-448,448).to(torch.float8_e4m3fn).float()
            v=v.float().clamp(-448,448).to(torch.float8_e4m3fn).float()
        key_cache.append(k);value_cache.append(v)
        index_projection=F.linear(x,i_weight)
        iq=rope(norm(index_projection[:index_heads*index_dim].reshape(index_heads,index_dim),iq_norm),coordinates)
        pending.append(index_projection[index_heads*index_dim:])
        if (step+1)%ratio==0:
            pooled=torch.stack(pending[-ratio:]).float().mean(0)
            if not fused:pooled=pooled.to(torch.bfloat16)
            compressed.append(rope(norm(pooled,ik_norm),positions[step+1-ratio]))
        if compressed:
            scores=torch.relu(iq.float()@torch.stack(compressed).float().T).sum(0)/index_dim**.5
            order=torch.topk(scores,min(len(compressed),config["indexer_budget"]//ratio)).indices.tolist()
        else:order=[]
        selected=[block*ratio+offset for block in order for offset in range(ratio)]+list(range((step+1)//ratio*ratio,step+1))
        selections.append(selected)
        cached_k=torch.stack(key_cache)[selected].repeat_interleave(heads//kv_heads,dim=1)
        cached_v=torch.stack(value_cache)[selected].repeat_interleave(heads//kv_heads,dim=1)
        scores=torch.einsum("hd,nhd->hn",q.float(),cached_k.float())/dim**.5
        probabilities=torch.softmax(scores,-1)
        if fresh:
            indices=torch.full((step+1,step+1),-1,dtype=torch.int32,device="cuda")
            for row,selection in enumerate(selections):indices[row,:len(selection)]=torch.tensor(selection,dtype=torch.int32,device="cuda")
            attended=sparse_gqa_fwd_interface_triton(torch.stack(queries).cuda().contiguous(),torch.stack(fresh_keys).cuda().contiguous(),
                    torch.stack(fresh_values).cuda().contiguous(),step+1,indices,torch.tensor([0,step+1],dtype=torch.int32,device="cuda"),dim**-.5)[-1].cpu()
        else:attended=torch.einsum("hn,nhd->hd",probabilities,cached_v.float()).to(torch.bfloat16)
        if fused:
            gated=fused_sigmoid_mul(attended.flatten().unsqueeze(0).cuda(),gate.unsqueeze(0).cuda(),inplace=False).flatten().cpu()
        else:gated=attended*torch.sigmoid(gate)
        expected.append(F.linear(gated.flatten(),o_weight).float())
    with tempfile.TemporaryDirectory(prefix="jitllm-qsa-") as temp:
        request=Path(temp)/"request.json"
        request.write_text(json.dumps({"mode":"qsa","inputs":[x.float().tolist() for x in inputs],"positions":positions,"kv_dtype":"fp8_e4m3" if fp8 else "bf16","fused_precision":fused,"fresh_prefill":fresh,"gpu_projections":projected,"device":device}))
        command=[sys.executable,str(ROOT/"jitllm_runtime.py"),"qsa-probe",str(model),str(request)] if gpu else [
                "java","-Xmx1g","-cp",str(ROOT/".local/jitllm/target/classes"),"org.beehive.jitllm.integration.cli.FlashNextLayerProbe",str(model),str(request)]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        lines=[line for line in result.stdout.splitlines() if line.startswith('{"')]
        if len(lines)!=1:raise RuntimeError("Expected one QSA report: "+result.stdout)
        actual=json.loads(lines[0])
    comparisons=[]
    for step,(reference,values,selection) in enumerate(zip(expected,actual["outputs"],actual["selections"],strict=True)):
        assert sorted(selection)==sorted(selections[step]),(step,selection,selections[step])
        observed=torch.tensor(values);differences=observed-reference
        torch.testing.assert_close(observed,reference,rtol=.016,atol=.004,msg=f"QSA step {step}")
        comparisons.append({"step":step,"max_absolute_error":differences.abs().max().item(),"rmse":differences.square().mean().sqrt().item(),"exact_fraction":(differences==0).float().mean().item(),"selected_tokens":len(selection)})
    report={"checkpoint":str(model.resolve()),"fused_precision":fused,"fresh_prefill":fresh,"attention_reference":"SGLang CUDA sparse GQA prefill" if fresh else "PyTorch FP32 attention", "gate_reference":"SGLang CUDA fused_sigmoid_mul" if fused else "PyTorch eager BF16 sigmoid/mul", "kv_dtype":"fp8_e4m3" if fp8 else "bf16","steps":len(inputs),"positions":positions,"compression_ratio":ratio,"token_budget":config["indexer_budget"],
            "rollback_exact":actual["rollback_exact"],"reset_exact":actual["reset_exact"],"cache_resident_bytes":actual["cache_resident_bytes"],"rtol":.016,"atol":.004,"comparisons":comparisons}
    if gpu:
        assert actual["gpu_qsa"]["kernel_ns"]>0 and actual["gpu_qsa"]["kv_host_download_bytes"]==0 and actual["gpu_qsa"]["reserved_bytes_after_close"]==0
        report["gpu_qsa"]=actual["gpu_qsa"];report["gpu_projections"]=projected
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path);parser.add_argument("--fp8-kv",action="store_true");parser.add_argument("--fused-precision",action="store_true");parser.add_argument("--fresh-prefill",action="store_true")
    parser.add_argument("--gpu-qsa",action="store_true");parser.add_argument("--gpu-projections",action="store_true");parser.add_argument("--device",type=int,default=1)
    args=parser.parse_args();report=json.dumps(verify(args.model,args.fp8_kv,args.fused_precision,args.fresh_prefill,args.gpu_qsa,args.gpu_projections,args.device),indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(report+"\n")
    print(report)

"""Validate native Gated DeltaNet recurrence and causal history with real checkpoint weights."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "model.language_model.layers.0.linear_attn."


def verify(model):
    torch.set_num_threads(4)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    def weight(name):
        name = PREFIX + name
        with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    config = json.loads((model / "config.json").read_text())["text_config"]
    heads, value_heads = config["linear_num_key_heads"], config["linear_num_value_heads"]
    key_dim, value_dim = config["linear_key_head_dim"], config["linear_value_head_dim"]
    width = heads*key_dim
    qkv_weight, z_weight = weight("in_proj_qkv.weight"), weight("in_proj_z.weight")
    a_weight, b_weight, out_weight = weight("in_proj_a.weight"), weight("in_proj_b.weight"), weight("out_proj.weight")
    conv, norm = weight("conv1d.weight"), weight("norm.weight")
    a_log, dt_bias = weight("A_log").float(), weight("dt_bias").float()
    indices = torch.arange(config["hidden_size"], dtype=torch.float32)
    inputs = [(torch.sin(indices*.015+step*.14)+.1*torch.cos(indices*.001-step*.1)).to(torch.bfloat16) for step in range(12)]
    conv_state = torch.zeros(qkv_weight.shape[0], config["linear_conv_kernel_dim"]-1)
    temporal = torch.zeros(value_heads,key_dim,value_dim)
    expected = []
    for x in inputs:
        projected = F.linear(x,qkv_weight)
        z = F.linear(x,z_weight)
        a, b = F.linear(x,a_weight).float(), F.linear(x,b_weight).float()
        conv_input = torch.cat([conv_state,projected.float().unsqueeze(-1)],dim=-1)
        # Native fused causal convolution keeps its accumulator in FP32 through SiLU.
        convolved = F.silu(F.conv1d(conv_input.unsqueeze(0),conv.float(),groups=qkv_weight.shape[0])).flatten().to(torch.bfloat16)
        conv_state = conv_input[:,1:]
        q = convolved[:width].float().reshape(heads,key_dim)
        k = convolved[width:2*width].float().reshape(heads,key_dim)
        v = convolved[2*width:].float().reshape(value_heads,value_dim)
        q = q * torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6) * key_dim**-.5
        k = k * torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        q = q.repeat_interleave(value_heads//heads,dim=0)
        k = k.repeat_interleave(value_heads//heads,dim=0)
        decay = (-torch.exp(a_log)*F.softplus(a+dt_bias)).exp()
        beta = torch.sigmoid(b)
        temporal = temporal*decay[:,None,None]
        correction = (v-torch.einsum("hkv,hk->hv",temporal,k))*beta[:,None]
        temporal = temporal + k[:,:,None]*correction[:,None,:]
        core = torch.einsum("hkv,hk->hv",temporal,q).to(torch.bfloat16)
        normalized = core.float()*torch.rsqrt(core.float().square().mean(-1,keepdim=True)+config["rms_norm_eps"])
        assert config["output_gate_type"]=="sigmoid"
        normalized = (normalized*norm.float()*torch.sigmoid(z.float().reshape(value_heads,value_dim))).to(torch.bfloat16)
        expected.append(F.linear(normalized.flatten(),out_weight).float())
    with tempfile.TemporaryDirectory(prefix="jitllm-deltanet-") as temp:
        request = Path(temp)/"request.json"
        request.write_text(json.dumps({"mode":"gdn","inputs":[x.float().tolist() for x in inputs]}))
        result = subprocess.run(["java","-Xmx1g","-cp",str(ROOT/"target/classes"),
                                 "org.beehive.jitllm.integration.cli.FlashNextLayerProbe",str(model),str(request)],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        actual = json.loads(result.stdout)
    comparisons = []
    for step,(reference,values) in enumerate(zip(expected,actual["outputs"],strict=True)):
        observed = torch.tensor(values)
        differences = observed-reference
        comparisons.append({"step":step,"max_absolute_error":differences.abs().max().item(),
                            "rmse":differences.square().mean().sqrt().item(),"exact_fraction":(differences==0).float().mean().item()})
        torch.testing.assert_close(observed,reference,rtol=.008,atol=.001,msg=f"DeltaNet step {step}")
    assert actual["cache_resident_bytes"]<=1048576
    return {"checkpoint":str(model.resolve()),"steps":len(inputs),"key_heads":heads,"value_heads":value_heads,
            "head_mapping":"repeat_interleave","temporal_dtype":"F32","activation_dtype":"BF16","output_gate":"sigmoid", "rtol":.008,"atol":.001,
            "rollback_exact":actual["rollback_exact"],"reset_exact":actual["reset_exact"],"cache_resident_bytes":actual["cache_resident_bytes"],"comparisons":comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.model),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report+"\n")
    print(report)

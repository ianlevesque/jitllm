"""Compare Java NVFP4 activation packing with FlashInfer's CUDA quantizer."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch
from flashinfer.quantization import fp4_quantize

ROOT=Path(__file__).resolve().parent


def verify():
    torch.cuda.set_device(0)
    generator=torch.Generator().manual_seed(20261012)
    rows=torch.randn(48,2560,generator=generator).to(torch.bfloat16)
    rows[0].zero_()
    rows[1]=torch.linspace(-6,6,2560).to(torch.bfloat16)
    reports=[]
    for scale in [.0005,.002,.01,1.]:
        inputs=rows.cuda()
        packed,scales=fp4_quantize(inputs,torch.tensor([1/scale],device="cuda",dtype=torch.float32),is_sf_swizzled_layout=False)
        expected_values=packed.cpu().view(torch.uint8)
        expected_scales=scales.cpu().view(torch.uint8).reshape(rows.shape[0],-1)[:,:rows.shape[1]//16]
        with tempfile.TemporaryDirectory(prefix="jitllm-activation-") as temp:
            request=Path(temp)/"request.json"
            request.write_text(json.dumps({"activations":rows.float().tolist(),"global_scale":scale}))
            result=subprocess.run(["java","-Xmx1g","-cp",str(ROOT/".local/jitllm/target/classes"),
                                   "org.beehive.jitllm.integration.cli.FlashNextCheckpointProbe","unused",str(request)],capture_output=True,text=True)
            if result.returncode: raise RuntimeError(result.stderr)
            actual=json.loads(result.stdout)["outputs"]
        values=torch.tensor([row["values"] for row in actual],dtype=torch.uint8)
        sf=torch.tensor([row["scales"] for row in actual],dtype=torch.uint8)
        if not torch.equal(values,expected_values):
            mismatches=(values!=expected_values).nonzero()
            raise AssertionError(f"scale={scale}: {len(mismatches)} packed values differ, first={mismatches[:12].tolist()}")
        torch.testing.assert_close(sf,expected_scales,rtol=0,atol=0)
        reports.append({"global_scale":scale,"rows":rows.shape[0],"columns":rows.shape[1],"packed_bytes_checked":values.numel(),
                        "scale_bytes_checked":sf.numel(),"all_bytes_exact":True})
    return {"device":torch.cuda.get_device_name(),"input_dtype":"BF16","quantizer":"FlashInfer fp4_quantize CUDA", "cases":reports}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    report=json.dumps(verify(),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report+"\n")
    print(report)

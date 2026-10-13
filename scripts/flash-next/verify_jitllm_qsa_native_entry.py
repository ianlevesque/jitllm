"""Prove the exported XQA C ABI matches the staged decode API before Java integration."""
import argparse
import ctypes as C
import json
import hashlib
import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.xqa import get_xqa_module
from flashinfer.jit.xqa import gen_xqa_module
from pathlib import Path
import tvm_ffi
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
libffi=C.CDLL(str(Path(tvm_ffi.__file__).parent/'lib/libtvm_ffi.so'),mode=C.RTLD_GLOBAL)
torch.cuda.set_device(next(i for i in range(torch.cuda.device_count()) if '5090' in torch.cuda.get_device_name(i)))
args_spec=(torch.bfloat16,torch.bfloat16,64,256,12,False,torch.bfloat16,1,False)
op=get_xqa_module(*args_spec);lib=C.CDLL(str(gen_xqa_module(*args_spec).get_library_path()))
class Dev(C.Structure):_fields_=[('type',C.c_int32),('id',C.c_int32)]
class Dtype(C.Structure):_fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class Tensor(C.Structure):_fields_=[('data',C.c_void_p),('device',Dev),('ndim',C.c_int32),('dtype',Dtype),('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('offset',C.c_uint64)]
class V(C.Union):_fields_=[('i',C.c_int64),('d',C.c_double),('p',C.c_void_p)]
class Any(C.Structure):_fields_=[('type',C.c_int32),('pad',C.c_uint32),('v',V)]
func=lib.__tvm_ffi_xqa_wrapper;func.argtypes=[C.c_void_p,C.POINTER(Any),C.c_int32,C.POINTER(Any)];func.restype=C.c_int
setstream=libffi.TVMFFIEnvSetStream;setstream.argtypes=[C.c_int32,C.c_int32,C.c_void_p,C.POINTER(C.c_void_p)];setstream.restype=C.c_int
keep=[]
def arg(x):
 a=Any()
 if isinstance(x,torch.Tensor):
  code,bits=(4,16) if x.dtype==torch.bfloat16 else (1,8) if x.dtype==torch.uint8 else (0,32)
  shape=(C.c_int64*x.ndim)(*x.shape);strides=(C.c_int64*x.ndim)(*x.stride());t=Tensor(x.data_ptr(),Dev(2,x.device.index),x.ndim,Dtype(code,bits,1),shape,strides,0);keep.extend([shape,strides,t,x]);a.type=7;a.v.p=C.addressof(t)
 elif isinstance(x,bool):a.type=2;a.v.i=int(x)
 elif isinstance(x,int):a.type=1;a.v.i=x
 elif isinstance(x,float):a.type=3;a.v.d=x
 return a
sm=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
torch.manual_seed(1701)
reports=[]
for count in [6,2054]:
 q=torch.randn((1,24,256),device='cuda',dtype=torch.bfloat16);k=torch.randn((33,2,64,256),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);table=torch.arange(33,device='cuda',dtype=torch.int32).reshape(1,-1);seq=torch.tensor([count],device='cuda',dtype=torch.int32)
 workspace=torch.zeros(16<<20,device='cuda',dtype=torch.uint8);out=torch.empty_like(q)
 expected=trtllm_batch_decode_with_kv_cache(query=q,kv_cache=(k,v),workspace_buffer=workspace,block_tables=table,seq_lens=seq,max_seq_len=2112,bmm1_scale=.0625,bmm2_scale=1.,enable_pdl=False)
 torch.cuda.synchronize()
 call_args=[False,sm,2,0,1.,None,out.unsqueeze(1),1.,q.unsqueeze(1),None,k.transpose(1,2),v.transpose(1,2),None,None,table,2112,seq.unsqueeze(1),1,1.,None,1,None,None,workspace[:8<<20],workspace[8<<20:],False]
 a=(Any*len(call_args))(*[arg(x) for x in call_args]);result=Any();previous=C.c_void_p();assert setstream(2,torch.cuda.current_device(),torch.cuda.current_stream().cuda_stream,C.byref(previous))==0
 try:
  status=func(None,a,len(call_args),C.byref(result))
  if status:
   error=C.c_void_p();libffi.TVMFFIErrorMoveFromRaised(C.byref(error))
   class Bytes(C.Structure):_fields_=[('data',C.c_void_p),('size',C.c_size_t)]
   b=Bytes.from_address(error.value+40);message=C.string_at(b.data,b.size).decode()
   libffi.TVMFFIObjectDecRef.argtypes=[C.c_void_p];libffi.TVMFFIObjectDecRef(error)
   raise RuntimeError(message)

 finally:assert setstream(2,torch.cuda.current_device(),previous,None)==0
 torch.cuda.synchronize()
 report={'selected_tokens':count,'ffi_status':status,'exact':torch.equal(out,expected),'max_error':(out-expected).abs().max().item(),'workspace_bytes':workspace.numel()}
 assert status==0 and report['exact'];reports.append(report);print(report,flush=True);keep.clear()
summary={'passed':True,'scope':'Exported C ABI feasibility check on Torch-owned CUDA buffers. Java/TornadoVM integration remains required.','flashinfer_version':__import__('flashinfer').__version__,'device':torch.cuda.get_device_name(),'library':str(gen_xqa_module(*args_spec).get_library_path()),'library_sha256':hashlib.sha256(Path(gen_xqa_module(*args_spec).get_library_path()).read_bytes()).hexdigest(),'cases':reports}
if args.output:
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

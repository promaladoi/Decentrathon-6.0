"""Isolated GPU screen only: no model, no training, no admission claim."""
import importlib.util
import json
from pathlib import Path
import torch
import triton
from solution.dropout import real_dropout_gather_kernel,_scales,BLOCK

spec=importlib.util.spec_from_file_location('direct_dropout',Path(__file__).with_name('direct_dropout.py'))
direct=importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)
results=[]
for count in (384,768,1280):
    width,batch,sequence=1536,4,512
    x=torch.randn((count,width),device='cuda')
    rows=torch.arange(count,device='cuda').contiguous()
    def native():
        dummy=torch.empty((batch,sequence,width),device='cuda')
        discarded,full=torch.ops.aten.native_dropout.default(dummy,.05,True)
        del dummy,discarded
        out=torch.empty(x.shape,device=x.device,dtype=torch.float16)
        packed=torch.empty((triton.cdiv(x.numel(),8),),device=x.device,dtype=torch.uint8)
        real_dropout_gather_kernel[(triton.cdiv(x.numel(),BLOCK),)](
            x,full,rows,out,packed,x.numel(),batch*sequence,width,_scales(.05)[0],
            BLOCK,num_warps=4,enable_fp_fusion=False)
        return out,packed
    def candidate(): return direct.gather(x,rows,batch,sequence,.05,1,0,torch.float16)
    a=float(triton.testing.do_bench(native,warmup=30,rep=100,quantiles=[.5]))
    b=float(triton.testing.do_bench(candidate,warmup=30,rep=100,quantiles=[.5]))
    results.append(dict(real_rows=count,native_gpu_ms=a,candidate_gpu_ms=b,gpu_ratio=a/b))
record=dict(kind='isolated GPU launch-chain timing; fixed explicit candidate RNG coordinates',
            full_task_admission=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            triton=triton.__version__,results=results)
print('DIRECT_DROPOUT_SCREEN',json.dumps(record),flush=True)
Path('direct_dropout_screen.json').write_text(json.dumps(record,indent=2)+'\n')

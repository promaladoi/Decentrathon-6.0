"""Inactive native batched gate/up screen; no training or admission claim."""
import json
from pathlib import Path
import torch
import triton

torch.manual_seed(702)
results=[]
for rows in (384,768,1280):
    x=torch.randn((rows,1536),device='cuda',dtype=torch.float16)
    weights=torch.randn((2,8960,1536),device='cuda',dtype=torch.float16)*.02
    def separate():
        return torch.nn.functional.linear(x,weights[0]),torch.nn.functional.linear(x,weights[1])
    def batched():
        return torch.bmm(x.unsqueeze(0).expand(2,-1,-1),weights.transpose(1,2))
    expected=separate()
    actual=batched()
    equal=all(torch.equal(actual[i],expected[i]) for i in range(2))
    max_error=max(float((actual[i]-expected[i]).abs().max()) for i in range(2))
    a=float(triton.testing.do_bench(separate,warmup=50,rep=200,quantiles=[.5])[0])
    b=float(triton.testing.do_bench(batched,warmup=50,rep=200,quantiles=[.5])[0])
    result=dict(real_rows=rows,bit_exact=equal,max_abs_error=max_error,
                separate_gpu_ms=a,batched_gpu_ms=b,gpu_ratio=a/b,
                extra_live_decoded_weight_bytes=8960*1536*2)
    results.append(result)
    print('NATIVE_GATE_UP_BMM',json.dumps(result),flush=True)
record=dict(kind='isolated native matmul screen; excludes NF4 decode and whole-command startup',
            full_task_admission=False,production_solution_changed=False,results=results)
Path('gate_up_bmm_screen.json').write_text(json.dumps(record,indent=2)+'\n')

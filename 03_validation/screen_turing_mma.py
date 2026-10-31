"""Finite v52 inactive MMA screen. No model training or production patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists(), 'preserve prior screens'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip(), 'both GPUs must be idle'
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'02_kernel_research'))
    import torch
    import torch.nn.functional as F
    from turing_mma import explicit_linear
    report=dict(version='v52',kind='inactive explicit-MMA mapping/performance research',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                production_solution_changed=False,full_task_admission=False,target_27_seconds_claimed=False,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cases=[],timing=[])
    try:
        for m,n,k in ((1,1,1),(16,8,8),(17,9,13),(63,31,33),(65,35,64),(128,128,256)):
            torch.manual_seed(5201)
            x=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
            w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.03
            actual,compiled=explicit_linear(x,w,fp32_output=True)
            torch.cuda.synchronize()
            reference=F.linear(x.double(),w.double()).float()
            native=F.linear(x,w)
            error=(actual-reference).abs()
            native_error=(actual.half()-native).abs()
            ptx=compiled.asm['ptx']
            mma=bool(re.search(r'\bmma\.sync\.aligned\.m16n8k8\.',ptx))
            row=dict(m=m,n=n,k=k,tensor_core_mma_found=mma,
                     max_abs_vs_fp64=float(error.max()),
                     relative_l2_vs_fp64=float(error.norm()/reference.norm().clamp_min(1e-12)),
                     half_max_abs_vs_native=float(native_error.max()),
                     half_exact_native=bool(torch.equal(actual.half(),native)))
            report['cases'].append(row)
            print('TURING_MMA_PROOF',json.dumps(row),flush=True)
            if not mma or row['relative_l2_vs_fp64'] > 1e-5:
                raise RuntimeError('MMA instruction/mapping numerical gate failed')
            (a.output.parent/'v52_mma.ptx').write_text(ptx)
        # A simple prototype is timed before any integration work. Timing does
        # not waive native arithmetic/gradient or original full-task checks.
        for m in (512,832):
            x=torch.randn(m,1536,device='cuda',dtype=torch.float16)*.1
            w=torch.randn(8960,1536,device='cuda',dtype=torch.float16)*.03
            actual,compiled=explicit_linear(x,w)
            reference=F.linear(x,w)
            error=(actual-reference).abs()
            qualified=bool(torch.allclose(actual,reference,rtol=2e-3,atol=3e-5))
            samples={'native':[],'prototype':[]}
            for _ in range(3):
                F.linear(x,w); explicit_linear(x,w)
            for repeat in range(6):
                for name in (('native','prototype') if repeat%2 else ('prototype','native')):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(3):
                        F.linear(x,w) if name=='native' else explicit_linear(x,w)
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end)/3)
            row=dict(tokens=m,native_close=qualified,max_abs_vs_native=float(error.max()),
                     median_ms={key:statistics.median(values) for key,values in samples.items()},samples_ms=samples)
            report['timing'].append(row)
            print('TURING_MMA_TIMING',json.dumps(row),flush=True)
        report['status']='screen_complete_not_integrated'
    except Exception as exc:
        report.update(status='research_rejected',error=type(exc).__name__+': '+str(exc))
        print('TURING_MMA_STOP',report['error'],flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(a.output,a.output.name)
        if (a.output.parent/'v52_mma.ptx').is_file(): z.write(a.output.parent/'v52_mma.ptx','v52_mma.ptx')
        for name in ('02_kernel_research/turing_mma.py','03_validation/screen_turing_mma.py'):
            z.write(repo/name,name)
    print('V52_SCREEN_ARCHIVE',json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
          status=report['status'])),flush=True)
    return 0 if report['status']=='screen_complete_not_integrated' else 2


if __name__=='__main__': raise SystemExit(main())

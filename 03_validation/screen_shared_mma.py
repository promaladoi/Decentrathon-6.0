"""Finite v57 shared-MMA numerical/performance screen, production unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--version',default='v57')
    a=p.parse_args()
    assert not a.output.exists() and not a.output.with_suffix('.zip').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip(), 'both GPUs must be idle'
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'02_kernel_research'))
    import torch
    import torch.nn.functional as F
    from shared_turing_mma import shared_linear
    report=dict(version=a.version,kind='inactive shared-memory SM75 MMA research',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        production_solution_changed=False,full_task_admission=False,target_27_seconds_claimed=False,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cases=[],timing=[])
    try:
        torch.manual_seed(5701)
        configurations=((128,128,8),(256,128,8),(256,64,8))
        for bm,bn,warps in configurations:
            for m,n,k in ((1,1,32),(17,9,64),(63,65,96),(129,137,256)):
                x=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
                w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.03
                actual,compiled=shared_linear(x,w,block_m=bm,block_n=bn,warps=warps,fp32_output=True)
                reference=F.linear(x.double(),w.double()).float()
                native=F.linear(x,w)
                ptx=compiled.asm['ptx']
                row=dict(m=m,n=n,k=k,block_m=bm,block_n=bn,warps=warps,
                    tensor_core_mma_found=bool(re.search(r'\bmma\.sync\.aligned\.m16n8k8\.',ptx)),
                    static_shared_declared=bool(re.search(r'\.shared\s+\.align\s+16\s+\.b8',ptx)),
                    registers=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),
                    max_abs_vs_fp64=float((actual-reference).abs().max()),
                    relative_l2_vs_fp64=float((actual-reference).norm()/reference.norm().clamp_min(1e-12)),
                    half_exact_native=bool(torch.equal(actual.half(),native)))
                report['cases'].append(row)
                print('SHARED_MMA_PROOF',json.dumps(row),flush=True)
                if not row['tensor_core_mma_found'] or row['relative_l2_vs_fp64'] > 1e-5:
                    raise RuntimeError('shared mapping / numerical gate failed')
                a.output.with_suffix('.ptx').write_text(ptx)
        for m in (512,832):
            x=torch.randn(m,1536,device='cuda',dtype=torch.float16)*.1
            w=torch.randn(8960,1536,device='cuda',dtype=torch.float16)*.03
            native=F.linear(x,w)
            for bm,bn,warps in configurations:
                for gm in (1,8):
                    actual,compiled=shared_linear(x,w,block_m=bm,block_n=bn,warps=warps,group_m=gm)
                    error=(actual-native).abs()
                    samples={'native':[],'prototype':[]}
                    functions=dict(native=lambda:F.linear(x,w),prototype=lambda:shared_linear(x,w,block_m=bm,block_n=bn,warps=warps,group_m=gm))
                    for _ in range(3):
                        for fn in functions.values(): fn()
                    for repeat in range(6):
                        for name in (list(functions) if repeat%2 else list(reversed(functions))):
                            start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record()
                            for _ in range(3): functions[name]()
                            end.record();end.synchronize()
                            samples[name].append(start.elapsed_time(end)/3)
                    row=dict(m=m,block_m=bm,block_n=bn,warps=warps,group_m=gm,
                        native_close=bool(torch.allclose(actual,native,rtol=2e-3,atol=3e-5)),
                        max_abs_vs_native=float(error.max()),
                        registers=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),
                        median_ms={key:statistics.median(v) for key,v in samples.items()},samples_ms=samples)
                    report['timing'].append(row)
                    print('SHARED_MMA_TIMING',json.dumps(row),flush=True)
        report['status']='screen_passed_not_integrated' if any(r['native_close'] and
            r['median_ms']['prototype'] < .95*r['median_ms']['native'] for r in report['timing']) else 'research_rejected_performance'
    except Exception as exc:
        report.update(status='research_rejected',error=type(exc).__name__+': '+str(exc))
        print('SHARED_MMA_STOP',report['error'],flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(a.output,a.output.name)
        if a.output.with_suffix('.ptx').is_file(): z.write(a.output.with_suffix('.ptx'),a.output.with_suffix('.ptx').name)
        for name in ('02_kernel_research/shared_turing_mma.py','03_validation/screen_shared_mma.py'):
            z.write(repo/name,name)
    print('SHARED_MMA_ARCHIVE',json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),status=report['status'])),flush=True)
    return 0 if report['status']=='screen_passed_not_integrated' else 2


if __name__=='__main__': raise SystemExit(main())

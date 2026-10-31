"""v61 inactive launch-overhead diagnostic; CUDA graphs are for timing only.

No graph, library observer, task or training implementation is changed. Compare
the same native/shared operations eagerly and through warmed CUDA graph replay
to identify CPU launch gaps in the earlier small-operation event timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    assert not a.output.exists() and not a.output.with_suffix('.zip').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip(), 'both GPUs must be idle'
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'02_kernel_research'))
    import torch
    import torch.nn.functional as F
    from shared_turing_mma import assembly, shared_linear
    report=dict(version='v61',kind='inactive eager vs graph-replay launch diagnostic',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        production_solution_changed=False,full_task_admission=False,target_27_seconds_claimed=False,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,rows=[])
    try:
        for m in (512,832):
            torch.manual_seed(6101)
            x=torch.randn(m,1536,device='cuda',dtype=torch.float16)*.1
            w=torch.randn(8960,1536,device='cuda',dtype=torch.float16)*.03
            functions=dict(native=lambda:F.linear(x,w),prototype=lambda:shared_linear(
                x,w,block_m=128,block_n=128,warps=8,group_m=8)[0])
            outputs={name:fn() for name,fn in functions.items()}
            assert torch.equal(outputs['native'],outputs['prototype'])
            graphs={}
            retained={}
            for name,fn in functions.items():
                for _ in range(3): fn()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4): retained[name]=fn()
                graph.replay()
                torch.cuda.synchronize()
                graphs[name]=graph
            assert torch.equal(retained['native'],retained['prototype'])
            samples={mode:{name:[] for name in functions} for mode in ('eager_event_ms','eager_wall_ms','graph_event_ms')}
            for repeat in range(6):
                for name in (list(functions) if repeat%2 else list(reversed(functions))):
                    torch.cuda.synchronize()
                    start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                    began=time.perf_counter()
                    start.record()
                    for _ in range(3): functions[name]()
                    end.record();end.synchronize()
                    samples['eager_wall_ms'][name].append((time.perf_counter()-began)*1000/3)
                    samples['eager_event_ms'][name].append(start.elapsed_time(end)/3)
                    start.record()
                    for _ in range(10): graphs[name].replay()
                    end.record();end.synchronize()
                    samples['graph_event_ms'][name].append(start.elapsed_time(end)/40)
            row=dict(m=m,assembly_characters=len(assembly(128,128,8,False)),
                sampled_native_exact=True,samples=samples,
                medians={mode:{name:statistics.median(values) for name,values in bucket.items()}
                    for mode,bucket in samples.items()})
            report['rows'].append(row)
            print('LAUNCH_DIAGNOSTIC',json.dumps(row),flush=True)
        report['status']='diagnostic_complete_not_integrated'
    except Exception as exc:
        report.update(status='diagnostic_failed',error=type(exc).__name__+': '+str(exc))
        print('LAUNCH_DIAGNOSTIC_STOP',report['error'],flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(a.output,a.output.name)
        for name in ('02_kernel_research/shared_turing_mma.py','03_validation/screen_shared_launch.py'):
            z.write(repo/name,name)
    print('LAUNCH_DIAGNOSTIC_ARCHIVE',json.dumps(dict(path=str(archive),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),status=report['status'])),flush=True)
    return 0 if report['status']=='diagnostic_complete_not_integrated' else 2


if __name__=='__main__': raise SystemExit(main())

"""v62 exact NF4 launch-geometry screen; selected v45 stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    assert not a.output.exists() and not a.output.with_suffix('.zip').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip(), 'both GPUs must be idle'
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo))
    import torch
    import bitsandbytes.functional as bnbf
    import triton
    from solution.nf4 import nf4_dequant_kernel
    report=dict(version='v62',kind='inactive exact NF4 block/warp screen',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        production_solution_changed=False,full_task_admission=False,target_27_seconds_claimed=False,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,rows=[])
    configs=((512,4),(1024,4),(2048,4),(4096,4),(8192,4),(1024,8),(2048,8),(4096,8),(8192,8))
    try:
        for shape in ((37,65),(256,1536),(1536,1536),(8960,1536)):
            torch.manual_seed(6201)
            dense=torch.randn(shape,device='cuda',dtype=torch.float16)*.03
            packed,state=bnbf.quantize_4bit(dense,blocksize=64,quant_type='nf4',compress_statistics=True)
            del dense
            reference=bnbf.dequantize_4bit(packed,state)
            output=torch.empty(shape,device='cuda',dtype=torch.float16)
            count=output.numel()
            compiled={}
            def launch(block,warps):
                return nf4_dequant_kernel[(triton.cdiv(count,block),)](
                    packed,state.absmax,state.code,state.state2.absmax,state.state2.code,state.offset,
                    output,count,64,True,256,block,num_warps=warps,enable_fp_fusion=False)
            for block,warps in configs:
                compiled[block,warps]=launch(block,warps)
                torch.testing.assert_close(output,reference,rtol=0,atol=0)
            # A graph here only amortizes diagnostic launch gaps, with no
            # change to the production dispatcher or original observers.
            graphs={}
            for block,warps in configs:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): launch(block,warps)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):
                    for _ in range(4): launch(block,warps)
                graphs[block,warps]=g
            samples={config:[] for config in configs}
            for repeat in range(8):
                order=configs if repeat%2 else tuple(reversed(configs))
                for config in order:
                    start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(10): graphs[config].replay()
                    end.record();end.synchronize()
                    samples[config].append(start.elapsed_time(end)/40)
            for config in configs:
                kernel=compiled[config]
                row=dict(shape=shape,block=config[0],warps=config[1],native_exact=True,
                    registers=getattr(kernel,'n_regs',None),spills=getattr(kernel,'n_spills',None),
                    kernel_graph_median_ms=statistics.median(samples[config]),samples_ms=samples[config])
                report['rows'].append(row)
                print('NF4_GEOMETRY',json.dumps(row),flush=True)
            del graphs,compiled,packed,state,output,reference
        report['status']='screen_complete_not_integrated'
    except Exception as exc:
        report.update(status='screen_failed',error=type(exc).__name__+': '+str(exc))
        print('NF4_GEOMETRY_STOP',report['error'],flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(a.output,a.output.name)
        for name in ('solution/nf4.py','03_validation/screen_nf4_launch_geometry.py'):
            z.write(repo/name,name)
    print('NF4_GEOMETRY_ARCHIVE',json.dumps(dict(path=str(archive),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),status=report['status'])),flush=True)
    return 0 if report['status']=='screen_complete_not_integrated' else 2


if __name__=='__main__': raise SystemExit(main())

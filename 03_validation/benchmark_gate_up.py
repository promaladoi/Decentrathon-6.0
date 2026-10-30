"""Finite, independent T4 screen; never replaces a full-task measurement."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/codex_gate_up_diagnostic.json'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous diagnostic')
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU process is active')
    sys.path.insert(0,str(repo/'02_kernel_research'))
    sys.path.insert(0,str(repo))
    import torch
    import triton
    assert torch.cuda.get_device_capability()==(7,5)
    from gate_up_forward import compiled_backend_probe
    ptx = compiled_backend_probe()
    backend = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   triton=triton.__version__,
                   tensor_core_mma_found=bool(re.search(r'\b(?:mma\.sync|wmma\.mma)\.',ptx)))
    args.output.with_suffix('.ptx').write_text(ptx)
    print('GATE_UP_BACKEND',json.dumps(backend),flush=True)
    if not backend['tensor_core_mma_found']:
        args.output.write_text(json.dumps(dict(
            kind='isolated research; not a full-task result', backend=backend,
            status='rejected_before_numerical_sweep', rows=[],
            reason='Candidate tl.dot did not generate Tensor Core MMA on this T4; keep native GEMMs'),indent=2))
        print('Research rejected: scalar-FMA lowering is unsuitable for this Tensor Core candidate.',flush=True)
        return 2
    # A failed strict research test stops the screen, without modifying its oracle.
    subprocess.run([sys.executable,'-m','pytest','02_kernel_research/test_gate_up_forward.py','-q'],
                   cwd=repo,check=True)
    import triton.testing
    from gate_up_forward import CONFIGS,fused_gate_up,production_gate_up
    rows=[]
    for m in (512,832,1024):
        torch.manual_seed(42)
        x=torch.randn(m,1536,device='cuda',dtype=torch.float16)*.1
        wg,wu=[torch.randn(8960,1536,device='cuda',dtype=torch.float16)*.03 for _ in range(2)]
        native_ms=triton.testing.do_bench(lambda:production_gate_up(x,wg,wu),warmup=20,rep=100)
        for config in CONFIGS:
            # Warm compile in this diagnostic process only; not a contest score.
            fused_gate_up(x,wg,wu,config)
            measured=triton.testing.do_bench(lambda:fused_gate_up(x,wg,wu,config),warmup=20,rep=100)
            row=dict(tokens=m,config=config,native_ms=native_ms,fused_ms=measured,speedup=native_ms/measured)
            rows.append(row)
            print('GATE_UP_DIAGNOSTIC',json.dumps(row),flush=True)
        del x,wg,wu
    args.output.write_text(json.dumps(dict(kind='isolated research; not a full-task result',backend=backend,rows=rows),indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())

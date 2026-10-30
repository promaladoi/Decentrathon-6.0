"""Finite, independent T4 screen; never replaces a full-task measurement."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU process is active')
    # A failed strict research test stops the screen, without modifying its oracle.
    subprocess.run([sys.executable,'-m','pytest','02_kernel_research/test_gate_up_forward.py','-q'],
                   cwd=repo,check=True)
    sys.path.insert(0,str(repo/'02_kernel_research'))
    sys.path.insert(0,str(repo))
    import torch
    import triton.testing
    from gate_up_forward import CONFIGS,fused_gate_up,production_gate_up
    assert torch.cuda.get_device_capability()==(7,5)
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
    dest=Path('/kaggle/working/codex_gate_up_diagnostic.json')
    if dest.exists(): raise FileExistsError('Preserve previous diagnostic')
    dest.write_text(json.dumps(dict(kind='isolated research; not a full-task result',rows=rows),indent=2))


if __name__=='__main__': main()

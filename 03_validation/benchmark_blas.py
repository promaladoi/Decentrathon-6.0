"""Compare PyTorch's vendor BLAS choices separately from contest measurements."""
import json
from pathlib import Path
import subprocess


def main():
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Run only after all paired GPU work finishes')
    import torch
    import triton.testing
    assert torch.cuda.get_device_capability()==(7,5)
    assert torch.__version__.startswith('2.11.')
    choose=torch.backends.cuda.preferred_blas_library
    original=choose()
    cases=[]
    for m in (512,832,1024):
        cases.extend([(f'gate_forward_{m}',m,8960,1536,True),
                      (f'down_forward_{m}',m,1536,8960,True),
                      (f'mlp_backward_{m}',m,1536,8960,False)])
    cases.extend([('head_forward',128,151936,1536,True),
                  ('head_backward',128,1536,151936,False)])
    rows=[]
    try:
        for name,m,n,k,transpose in cases:
            torch.manual_seed(17)
            a=torch.randn((m,k),device='cuda',dtype=torch.float16)*.03
            b=torch.randn((n,k) if transpose else (k,n),device='cuda',dtype=torch.float16)*.03
            weight=b.t() if transpose else b
            fn=lambda: torch.mm(a,weight)
            choose(original)
            expected=fn()
            base=triton.testing.do_bench(fn,warmup=20,rep=80)
            choose('cublaslt')
            actual=fn()
            relative=(torch.linalg.vector_norm(actual.float()-expected.float())/
                      torch.linalg.vector_norm(expected.float())).item()
            if relative>=.001:
                raise AssertionError('Vendor choice exceeds FP16 global-L2 screening bound')
            alternate=triton.testing.do_bench(fn,warmup=20,rep=80)
            row=dict(operation=name,default_ms=base,cublaslt_ms=alternate,
                     speedup=base/alternate,relative_l2=relative)
            rows.append(row)
            print('BLAS_DIAGNOSTIC',json.dumps(row),flush=True)
            del a,b,weight,expected,actual
    finally:
        choose(original)
    result=dict(kind='isolated vendor-BLAS diagnostic, not a full-task result',
                default_backend=str(original),rows=rows,
                aggregate_speedup=sum(r['default_ms'] for r in rows)/sum(r['cublaslt_ms'] for r in rows))
    path=Path('/kaggle/working/codex_blas_diagnostic.json')
    if path.exists(): raise FileExistsError('Preserve the original diagnostic')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('BLAS_DIAGNOSTIC_COMPLETE',json.dumps({'aggregate_speedup':result['aggregate_speedup']}),flush=True)


if __name__=='__main__': main()

"""Separate native-vs-own two-product GPU diagnostic; never a contest score."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo))
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Wait for all measured GPU jobs to finish')
    import torch
    import triton
    from solution.fused_mlp_gradient import fused_gradient
    assert torch.cuda.get_device_capability()==(7,5)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    rows=[]
    for m in (256,512,832,1024):
        torch.manual_seed(17)
        dg,du=[torch.randn((m,8960),device='cuda',dtype=torch.float16)*.1 for _ in range(2)]
        wg,wu=[torch.randn((8960,1536),device='cuda',dtype=torch.float16)*.1 for _ in range(2)]
        def native():
            out=(dg@wg).float()
            out+=(du@wu).float()
            return out
        expected=native()
        base=triton.testing.do_bench(native,warmup=50,rep=100)
        for config in ((32,64,32,4),(32,64,64,4),(64,64,64,4),(32,128,64,4),
                       (64,128,32,4),(64,128,64,8)):
            actual=fused_gradient(dg,du,wg,wu,config)
            error=(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)).item()
            if error>=.001:
                raise AssertionError('two-product numeric error exceeds bound')
            ms=triton.testing.do_bench(lambda:fused_gradient(dg,du,wg,wu,config),warmup=50,rep=100)
            row=dict(tokens=m,config=config,native_ms=base,candidate_ms=ms,speedup=base/ms,relative_l2=error)
            rows.append(row)
            print('MLP_GRAD_DIAGNOSTIC',json.dumps(row),flush=True)
    doc=dict(kind='standalone diagnostic, not admitted full-task timing',code_commit=code,rows=rows)
    report=Path('/kaggle/working/codex_mlp_gradient_diagnostic.json')
    if report.exists(): raise FileExistsError('Preserve previous diagnostic')
    report.write_text(json.dumps(doc,indent=2)+'\n')
    with zipfile.ZipFile(report.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        z.write(report,report.name)
        z.write(repo/'solution/fused_mlp_gradient.py','solution/fused_mlp_gradient.py')
    print('MLP_GRAD_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__': main()

"""Screen exact-value frozen-weight layouts; not a contest benchmark."""
import json
from pathlib import Path
import subprocess


def main():
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Another GPU process is active')
    import torch
    import triton.testing
    assert torch.cuda.get_device_capability() == (7, 5)
    rows = []
    for tokens in (512, 832, 1024):
        for name, n, k in [('gate_up',8960,1536), ('down',1536,8960), ('attention',1536,1536)]:
            torch.manual_seed(314)
            weight = torch.randn(n,k,device='cuda',dtype=torch.float16) * .03
            alternate = weight.t().contiguous().t()
            assert torch.equal(weight, alternate)
            x = torch.randn(tokens,k,device='cuda',dtype=torch.float16) * .03
            dy = torch.randn(tokens,n,device='cuda',dtype=torch.float16) * .03
            for direction, a, b, c in [('forward',x,weight.t(),alternate.t()),
                                        ('backward',dy,weight,alternate)]:
                reference = torch.mm(a,b)
                actual = torch.mm(a,c)
                relative = ((actual.float()-reference.float()).norm() / reference.float().norm()).item()
                base_ms = triton.testing.do_bench(lambda: torch.mm(a,b),warmup=20,rep=100)
                alt_ms = triton.testing.do_bench(lambda: torch.mm(a,c),warmup=20,rep=100)
                row = dict(tokens=tokens, matrix=name, direction=direction,
                           original_ms=base_ms, transposed_ms=alt_ms,
                           relative_l2=relative, speedup=base_ms/alt_ms)
                rows.append(row)
                print('LAYOUT_DIAGNOSTIC',json.dumps(row),flush=True)
            del weight,alternate,x,dy,a,b,c,reference,actual
    # Two gate/up pairs and one down pair, for every layer. Attention is a
    # separate diagnostic; this summary does not pretend to be full task time.
    weighted = [dict(r, multiplicity=2 if r['matrix']=='gate_up' else 1)
                for r in rows if r['matrix']!='attention']
    ratio = sum(r['original_ms']*r['multiplicity'] for r in weighted) / sum(
        r['transposed_ms']*r['multiplicity'] for r in weighted)
    result = dict(kind='isolated layout diagnostic; not a full-task score',
                  weighted_mlp_speedup=ratio, rows=rows)
    path=Path('/kaggle/working/codex_weight_layout.json')
    if path.exists():
        raise FileExistsError('Preserve prior diagnostic')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('LAYOUT_COMPLETE',json.dumps({'weighted_mlp_speedup':ratio}),flush=True)


if __name__=='__main__':
    main()

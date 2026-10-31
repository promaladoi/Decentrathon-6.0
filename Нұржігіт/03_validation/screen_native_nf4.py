"""Inactive v54 screen of the already-installed bitsandbytes NF4 dispatcher.

This is vendor baseline code, not a team kernel. It is deliberately outside
solution/ and cannot alter the selected training path or its admission.
Primary ABI: bitsandbytes 0.50.2, backends/cuda/ops.py,
_gemm_4bit_kernel_impl. Compare native dispatch, the selected team's fallback,
and an explicit call of that vendor's fused backend before considering any
integration. No extra kernel library is installed or copied.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix('.zip').exists()
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip(), 'both GPUs must be idle'
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.backends.cuda import ops
    from solution.nf4 import _linear_forward, _uses_native_custom_gemm

    assert bnb.__version__ == '0.50.2' and torch.cuda.get_device_capability() == (7, 5)
    implementation = ops._gemm_4bit_kernel_impl
    report = dict(version='v54', kind='inactive vendor-dispatch numerical/performance screen',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, bitsandbytes=bnb.__version__,
        vendor_implementation_sha256=hashlib.sha256(inspect.getsource(implementation).encode()).hexdigest(),
        production_solution_changed=False, full_task_admission=False,
        target_27_seconds_claimed=False, rows=[])

    def fused(x, weight, state):
        return implementation(x, weight, state.shape, state.state2.absmax,
            state.blocksize, state.quant_type, absmax_8bit=state.absmax,
            absmax_code=state.state2.code, absmax_offset=state.offset)

    try:
        for n, k in ((8960, 1536), (1536, 8960), (1536, 1536), (256, 1536)):
            torch.manual_seed(5401)
            dense = torch.randn(n, k, device='cuda', dtype=torch.float16) * 0.03
            weight, state = bnb.functional.quantize_4bit(dense, blocksize=64,
                compress_statistics=True, quant_type='nf4')
            del dense
            for m in (128, 512, 832):
                x = torch.randn(m, k, device='cuda', dtype=torch.float16) * 0.1
                reference = _linear_forward(x, weight, None, state)
                actual = fused(x, weight, state)
                native = bnb.matmul_4bit(x, weight.t(), quant_state=state)
                error = (actual.float() - reference.float()).abs()
                row = dict(m=m, n=n, k=k,
                    selected_dispatch_uses_fused=_uses_native_custom_gemm(x, state),
                    exact_selected=bool(torch.equal(actual, reference)),
                    exact_vendor_dispatch=bool(torch.equal(actual, native)),
                    max_abs_vs_selected=float(error.max()),
                    relative_l2_vs_selected=float(error.norm()/reference.float().norm().clamp_min(1e-12)),
                    close_selected=bool(torch.allclose(actual, reference, rtol=2e-3, atol=3e-5)))
                functions = dict(selected=lambda: _linear_forward(x, weight, None, state),
                                 fused_vendor=lambda: fused(x, weight, state))
                samples = {name: [] for name in functions}
                for _ in range(3):
                    for fn in functions.values(): fn()
                for repeat in range(6):
                    order = list(functions) if repeat % 2 else list(reversed(functions))
                    for name in order:
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(5): functions[name]()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end)/5)
                row['samples_ms'] = samples
                row['median_ms'] = {name: statistics.median(values) for name, values in samples.items()}
                # This allocation diagnostic is not the organizer's NVML peak.
                allocations = {}
                del reference, actual, native, error
                torch.cuda.synchronize()
                for name, fn in functions.items():
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    value = fn()
                    torch.cuda.synchronize()
                    allocations[name] = torch.cuda.max_memory_allocated() - before
                    del value
                row['temporary_allocation_bytes_not_contest_peak'] = allocations
                report['rows'].append(row)
                print('NATIVE_NF4_SCREEN', json.dumps(row), flush=True)
            del x, weight, state
        candidates = [r for r in report['rows'] if not r['selected_dispatch_uses_fused']
                      and r['close_selected'] and r['median_ms']['fused_vendor'] < .9*r['median_ms']['selected']]
        report['status'] = 'requires_integration_validation' if candidates else 'research_rejected_no_clear_gain'
        report['potential_cases'] = [dict(m=r['m'], n=r['n'], k=r['k']) for r in candidates]
    except Exception as exc:
        report.update(status='research_rejected', error=type(exc).__name__ + ': ' + str(exc))
        print('NATIVE_NF4_STOP', report['error'], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.write(args.output, args.output.name)
        z.write(__file__, '03_validation/screen_native_nf4.py')
    print('V54_SCREEN_ARCHIVE', json.dumps(dict(path=str(archive),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), status=report['status'])), flush=True)
    return 0 if report['status'] == 'requires_integration_validation' else 2


if __name__ == '__main__':
    raise SystemExit(main())

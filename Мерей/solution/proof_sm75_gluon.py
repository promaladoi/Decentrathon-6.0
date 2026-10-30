"""Tiny explicit-layout SM75 execution/PTX proof, not a training benchmark.

python solution/proof_sm75_gluon.py --out /kaggle/working/sm75_gluon_proof.json --dump-ir
Stop if PTX lacks Turing mma.sync.m16n8k8; do not tune the old scalar-FMA path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from solution.sm75_gluon import sm75_dense_matmul


def _timing(call, iterations):
    torch.cuda.synchronize()
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    first.record()
    for _ in range(iterations):
        value = call()
        del value
    last.record()
    last.synchronize()
    return {'iterations': iterations, 'cuda_ms_per_call': first.elapsed_time(last) / iterations,
            'wall_ms_per_call': 1000 * (time.perf_counter() - started) / iterations}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dump-ir', action='store_true')
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'sm75_gluon_explicit_layout_execution_proof_not_training_score',
              'torch': torch.__version__, 'triton': triton.__version__,
              'source_sha256': hashlib.sha256((ROOT / 'solution/sm75_gluon.py').read_bytes()).hexdigest(),
              'encoding': {'version': [2, 1], 'warps_per_cta': [2, 2], 'instr_shape': [16, 8]},
              'installed_compiler_modified': False, 'production_dispatch_enabled': False,
              'cases': []}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
            raise RuntimeError('Run this proof on the task Tesla T4 / SM75')
        report['device'] = torch.cuda.get_device_name(0)
        for m, n, k in [(32, 32, 32), (64, 64, 128), (65, 33, 71)]:
            # Exact small integer products distinguish layout/operand packing
            # errors from independent GEMM floating-point reduction order.
            a = ((torch.arange(m * k, device='cuda') % 5) - 2).half().reshape(m, k)
            b = ((torch.arange(k * n, device='cuda') % 7) - 3).half().reshape(k, n)
            expected = torch.matmul(a, b).float()
            before_rng = torch.cuda.get_rng_state()
            torch.cuda.synchronize()
            started = time.perf_counter()
            actual, compiled = sm75_dense_matmul(a, b, return_compiled=True)
            torch.cuda.synchronize()
            case = {'m': m, 'n': n, 'k': k,
                    'compile_and_first_call_wall_s': time.perf_counter() - started}
            report['cases'].append(case)
            ptx = compiled.asm.get('ptx', '')
            case['turing_mma_m16n8k8_count'] = len(re.findall(
                r'mma\.sync\.aligned\.m16n8k8\.row\.col\.f32\.f16\.f16\.f32', ptx))
            case['ampere_m16n8k16_count'] = ptx.count('mma.sync.aligned.m16n8k16')
            case['n_regs'] = getattr(compiled, 'n_regs', None)
            case['n_spills'] = getattr(compiled, 'n_spills', None)
            metadata = getattr(compiled, 'metadata', None)
            case['shared_bytes'] = getattr(metadata, 'shared', None)
            case['integer_values_exact'] = bool(torch.equal(actual, expected))
            case['rng_unchanged'] = bool(torch.equal(torch.cuda.get_rng_state(), before_rng))
            if args.dump_ir:
                directory = path.parent / (path.stem + '_ir')
                directory.mkdir(exist_ok=True)
                case['ir_paths'] = {}
                for kind in ('ptx', 'ttgir', 'ttir', 'llir'):
                    value = compiled.asm.get(kind)
                    if isinstance(value, str):
                        target = directory / (str(m) + 'x' + str(n) + 'x' + str(k) + '.' + kind)
                        target.write_text(value, encoding='utf-8')
                        case['ir_paths'][kind] = str(target)
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            assert case['turing_mma_m16n8k8_count'] > 0, 'No explicit Turing tensor-core instruction in PTX'
            assert case['ampere_m16n8k16_count'] == 0, 'Ampere-only instruction appeared on SM75'
            assert case['integer_values_exact'], 'Explicit MMA operand/output layout mismatch'
            assert case['rng_unchanged']
            for _ in range(3):
                sm75_dense_matmul(a, b)
            case['gluon_tiny_hot'] = _timing(lambda: sm75_dense_matmul(a, b), args.iterations)
            case['torch_half_mm_then_float_tiny_hot'] = _timing(
                lambda: torch.matmul(a, b).float(), args.iterations)
            print(json.dumps(case), flush=True)
        report['status'] = 'turing_mma_ptx_and_exact_integer_execution_proved'
    except Exception as error:
        report['status'] = 'proof_failed_do_not_enable'
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        raise
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SM75_GLUON_PROOF_COMPLETE', str(path), flush=True)


if __name__ == '__main__':
    main()

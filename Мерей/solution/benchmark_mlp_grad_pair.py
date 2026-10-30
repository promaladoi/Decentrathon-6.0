"""Standalone paired NF4-MLP diagnostic, not a training or admission score.

Run from Мерей: python solution/benchmark_mlp_grad_pair.py --out /kaggle/working/mlp_pair.json
Default: one tile and the real K=8960/N=1536 dimensions. Use --configs all
for an explicit sweep. No production patch or dispatch is installed.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.backward_sums import sum_half_gradients
from solution.mlp_grad_pair import CONFIGS, DEFAULT_CONFIG, mlp_grad_pair
from solution.nf4 import dequantize_nf4


def reference_parts(dg, du, gate_weight, up_weight):
    """The current Merei backward: decode on demand, two unchanged native GEMMs."""
    gate = torch.matmul(dg, dequantize_nf4(gate_weight, gate_weight.quant_state, torch.float16))
    up = torch.matmul(du, dequantize_nf4(up_weight, up_weight.quant_state, torch.float16))
    return gate, up


def reference_output(dg, du, gate_weight, up_weight, output_dtype):
    return sum_half_gradients(reference_parts(dg, du, gate_weight, up_weight)).to(output_dtype)


def accuracy_metrics(actual, expected, gate_part, up_part):
    """Retain the strict partial-ULP gate; do not hide cancellation outliers.

    This is an empirical qualification criterion, not a mathematical promise
    that an arbitrary GEMM reduction must match cuBLAS within two half ULPs.
    The other team's rejected research demonstrates why L2 alone is inadequate.
    """
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    same_special = (torch.equal(torch.isnan(a), torch.isnan(b))
                    and torch.equal(torch.isposinf(a), torch.isposinf(b))
                    and torch.equal(torch.isneginf(a), torch.isneginf(b)))
    difference = (a[finite] - b[finite]).abs()
    denominator = b[finite].norm().clamp(min=1e-12)
    relative_l2 = float(difference.norm() / denominator)

    def half_spacing(value):
        magnitude = value.abs()
        toward = torch.full_like(magnitude, float('inf'))
        return (torch.nextafter(magnitude, toward).float() - magnitude.float()).abs()

    budget = 2 * (half_spacing(gate_part) + half_spacing(up_part))
    # At a half maximum nextafter is Inf; those outputs are not a useful
    # finite-ULP oracle. Special classification and global L2 still apply.
    checked = finite & torch.isfinite(budget)
    violations = int(((a - b).abs()[checked] > budget[checked]).sum())
    return {'same_nan_and_signed_inf_locations': bool(same_special),
            'finite_relative_l2': relative_l2,
            'max_finite_abs_error': float(difference.max()) if difference.numel() else 0.0,
            'partial_two_ulp_checked_entries': int(checked.sum()),
            'partial_two_ulp_violations': violations,
            'bitwise_equal': bool(torch.equal(actual.view(torch.int16 if actual.dtype == torch.float16
                                                           else torch.int32),
                                               expected.view(torch.int16 if expected.dtype == torch.float16
                                                             else torch.int32))),
            'qualified': bool(same_special and relative_l2 <= 1e-3 and violations == 0)}


def _sample(call):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    output = call()
    end.record()
    end.synchronize()
    result = {'cuda_ms': begin.elapsed_time(end), 'wall_ms': 1000 * (time.perf_counter() - started)}
    del output
    return result


def _memory(call):
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    with_output = torch.cuda.memory_allocated()
    del output
    torch.cuda.synchronize()
    return {'start_allocated_bytes': before, 'peak_allocated_bytes': peak,
            'extra_allocated_peak_bytes': peak - before,
            'allocated_with_output_bytes': with_output,
            'allocated_after_release_bytes': torch.cuda.memory_allocated(),
            'meaning': 'one-call PyTorch allocated bytes, not whole-training NVML peak'}


def _summary(samples):
    return {key: {'median': statistics.median(row[key] for row in samples),
                  'min': min(row[key] for row in samples), 'max': max(row[key] for row in samples)}
            for key in ('cuda_ms', 'wall_ms')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', nargs='+', type=int, default=[512, 833, 1394])
    parser.add_argument('--configs', nargs='+', default=[DEFAULT_CONFIG])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--time-unqualified', action='store_true',
                        help='research-only timings remain explicitly marked accuracy-failed')
    args = parser.parse_args()
    configs = list(CONFIGS) if args.configs == ['all'] else args.configs
    if any(config not in CONFIGS for config in configs):
        parser.error('unknown config; use one of ' + ', '.join(CONFIGS))
    if min(args.rows) <= 0 or args.rounds < 1 or args.samples < 1:
        parser.error('rows, rounds and samples must be positive')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This diagnostic requires the task Tesla T4')
    import bitsandbytes.functional as bnb

    sources = {}
    for name in ['mlp_grad_pair.py', 'benchmark_mlp_grad_pair.py', 'nf4.py', 'backward_sums.py']:
        sources[name] = hashlib.sha256((ROOT / 'solution' / name).read_bytes()).hexdigest()
    report = {'kind': 'experimental_nf4_mlp_pair_not_training_score',
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'triton': triton.__version__, 'bitsandbytes': importlib.metadata.version('bitsandbytes'),
              'sources': sources, 'k': 8960, 'n': 1536, 'rounds': args.rounds,
              'samples_per_block': args.samples, 'cold_compile_counts_in_training': True,
              'dense_weight_cache': False, 'split_k_workspace': False,
              'reference': 'current Merei on-demand NF4 decode + two torch.matmul + sum_half_gradients',
              'ordering': 'each round: reference,candidate then candidate,reference', 'cases': []}
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    torch.manual_seed(954)
    weights = []
    for _ in range(2):
        dense = torch.randn((8960, 1536), device='cuda', dtype=torch.float16) * .02
        packed, state = bnb.quantize_4bit(dense, blocksize=64, compress_statistics=True, quant_type='nf4')
        packed.quant_state = state
        weights.append(packed)
        del dense
    gate_weight, up_weight = weights
    for rows in args.rows:
        dg = torch.randn((rows, 8960), device='cuda', dtype=torch.float16)
        du = torch.randn_like(dg)
        before_rng = torch.cuda.get_rng_state()
        for config in configs:
            case = {'m': rows, 'config': config, 'tile': CONFIGS[config]}
            report['cases'].append(case)
            candidate = lambda: mlp_grad_pair(dg, du, gate_weight, up_weight, config=config)
            reference = lambda: reference_output(dg, du, gate_weight, up_weight, torch.float32)
            try:
                torch.cuda.synchronize()
                started = time.perf_counter()
                actual = candidate()
                torch.cuda.synchronize()
                case['compile_plus_first_call_wall_ms'] = 1000 * (time.perf_counter() - started)
            except Exception as error:
                case['error'] = type(error).__name__ + ': ' + str(error)
                case['status'] = 'compile_or_resource_failure'
                save()
                if type(error).__name__ not in ('OutOfResources', 'CompilationError',
                                                'UnsupportedLanguageConstruct'):
                    raise
                continue
            gate_part, up_part = reference_parts(dg, du, gate_weight, up_weight)
            expected = sum_half_gradients((gate_part, up_part))
            case['accuracy'] = accuracy_metrics(actual, expected, gate_part, up_part)
            del actual, expected, gate_part, up_part
            if not case['accuracy']['qualified'] and not args.time_unqualified:
                case['status'] = 'accuracy_failed_not_timed'
                save()
                continue
            for _ in range(3):
                for call in (reference, candidate):
                    value = call()
                    del value
            samples = {'reference': [], 'candidate': []}
            for _ in range(args.rounds):
                for block in (('reference', reference), ('candidate', candidate),
                              ('candidate', candidate), ('reference', reference)):
                    name, call = block
                    samples[name].extend(_sample(call) for _ in range(args.samples))
            case['samples'] = samples
            case['summary'] = {name: _summary(values) for name, values in samples.items()}
            case['memory'] = {'reference': _memory(reference), 'candidate': _memory(candidate)}
            case['rng_unchanged'] = bool(torch.equal(torch.cuda.get_rng_state(), before_rng))
            case['status'] = 'qualified_microbenchmark_only' if case['accuracy']['qualified'] \
                else 'accuracy_failed_research_timings_only'
            print(json.dumps({k: v for k, v in case.items() if k != 'samples'}), flush=True)
            save()
        del dg, du
    report['completed'] = True
    save()
    print('MLP_PAIR_DIAGNOSTIC_COMPLETE', str(output_path), flush=True)


if __name__ == '__main__':
    main()

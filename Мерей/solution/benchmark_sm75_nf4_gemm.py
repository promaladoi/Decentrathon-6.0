"""Explicit-MMA SM75 NF4 diagnostic versus current on-demand decoder + Torch.

python solution/benchmark_sm75_nf4_gemm.py --rows 833 --tiles m32n64k32s1 --out /kaggle/working/sm75_nf4.json
No production patch is installed. First-call wall includes compilation only
when the caller gives a fresh cache; hot latency is reported separately.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from solution.nf4_gemm import reference_nf4_gemm
from solution.sm75_nf4_gemm import TILES, sm75_nf4_gemm


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
    del output
    torch.cuda.synchronize()
    return {'extra_peak_allocated_bytes': peak - before, 'start_allocated_bytes': before,
            'end_after_release_allocated_bytes': torch.cuda.memory_allocated(),
            'meaning': 'one-call PyTorch allocated peak, not full-run NVML peak'}


def accuracy(actual, expected):
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite] - b[finite]).abs()
    relative = float(delta.norm() / b[finite].norm().clamp(min=1e-12))
    same_special = (torch.equal(torch.isnan(a), torch.isnan(b))
                    and torch.equal(torch.isposinf(a), torch.isposinf(b))
                    and torch.equal(torch.isneginf(a), torch.isneginf(b)))
    # Keep explicit elementwise and global gates. Separate MMA reductions are
    # not promised bitwise equal to a particular cuBLAS reduction algorithm.
    close = bool(torch.all(delta <= (2e-3 + 2e-3 * b[finite].abs())))
    return {'same_nan_signed_inf': bool(same_special), 'finite_relative_l2': relative,
            'max_finite_abs_error': float(delta.max()) if delta.numel() else 0.0,
            'elementwise_rtol_0_002_atol_0_002': close,
            'bitwise_equal': bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16))),
            'qualified': bool(same_special and close and relative <= 1e-3)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', nargs='+', type=int, default=[833])
    parser.add_argument('--tiles', nargs='+', default=['m32n64k32s1'])
    parser.add_argument('--directions', nargs='+', choices=['forward', 'input_gradient'],
                        default=['forward', 'input_gradient'])
    parser.add_argument('--shapes', nargs='+', choices=['gate', 'down'], default=['gate', 'down'])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--time-unqualified', action='store_true')
    parser.add_argument('--dump-ir', action='store_true')
    args = parser.parse_args()
    tiles = list(TILES) if args.tiles == ['all'] else args.tiles
    if (min(args.rows) < 1 or args.rounds < 1 or args.samples < 1
            or any(tile not in TILES for tile in tiles)):
        parser.error('positive dimensions/counts and known tiles are required')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError('Run this experiment on the task SM75 Tesla T4')
    import bitsandbytes.functional as bnb

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'sm75_gluon_nf4_gemm_micro_not_training_score',
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'triton': triton.__version__, 'cache_dir': os.environ.get('TRITON_CACHE_DIR'),
              'first_call_may_include_compilation': True, 'dense_weight_cache': False,
              'split_k_workspace': False, 'production_dispatch': False,
              'reference': 'current Merei NF4 decode on demand plus torch.mm',
              'ordering': 'each round reference,candidate,candidate,reference', 'sources': {}, 'cases': []}
    for name in ['sm75_nf4_gemm.py', 'sm75_gluon.py', 'nf4.py', 'nf4_gemm.py',
                 'benchmark_sm75_nf4_gemm.py']:
        report['sources'][name] = hashlib.sha256((ROOT / 'solution' / name).read_bytes()).hexdigest()

    def save():
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    torch.manual_seed(985)
    shapes = {'gate': (8960, 1536), 'down': (1536, 8960)}
    for shape_name in args.shapes:
        shape = shapes[shape_name]
        dense = torch.randn(shape, device='cuda', dtype=torch.float16) * .02
        packed, state = bnb.quantize_4bit(dense, blocksize=64, compress_statistics=True, quant_type='nf4')
        del dense
        for direction in args.directions:
            transpose = direction == 'forward'
            for m in args.rows:
                a = torch.randn((m, shape[1 if transpose else 0]), device='cuda', dtype=torch.float16) * .2
                reference = lambda: reference_nf4_gemm(a, packed, state, transpose_weight=transpose)
                expected = reference()
                before_rng = torch.cuda.get_rng_state()
                for name in tiles:
                    tile = TILES[name]
                    call = lambda: sm75_nf4_gemm(a, packed, state, transpose_weight=transpose, tile=tile)
                    case = {'shape': list(shape), 'shape_name': shape_name, 'direction': direction,
                            'm': m, 'tile_name': name, 'tile': asdict(tile)}
                    report['cases'].append(case)
                    try:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        actual, compiled = sm75_nf4_gemm(a, packed, state, transpose_weight=transpose,
                                                         tile=tile, return_compiled=True)
                        torch.cuda.synchronize()
                        case['compile_and_first_call_wall_s'] = time.perf_counter() - started
                        ptx = compiled.asm.get('ptx', '')
                        case['mma_m16n8k8'] = len(re.findall(r'mma\.sync\.aligned\.m16n8k8', ptx))
                        case['mma_m16n8k16'] = ptx.count('mma.sync.aligned.m16n8k16')
                        case['n_regs'] = getattr(compiled, 'n_regs', None)
                        case['n_spills'] = getattr(compiled, 'n_spills', None)
                        case['shared_bytes'] = getattr(getattr(compiled, 'metadata', None), 'shared', None)
                        case['accuracy'] = accuracy(actual, expected)
                        del actual
                        if args.dump_ir:
                            directory = output_path.parent / (output_path.stem + '_ir')
                            directory.mkdir(exist_ok=True)
                            stem = shape_name + '_' + direction + '_' + name
                            for kind in ('ptx', 'ttgir'):
                                ir = compiled.asm.get(kind)
                                if isinstance(ir, str):
                                    (directory / (stem + '.' + kind)).write_text(ir, encoding='utf-8')
                        if case['mma_m16n8k8'] == 0 or case['mma_m16n8k16']:
                            case['status'] = 'wrong_hardware_path_not_timed'
                            save()
                            continue
                        if case['n_spills'] != 0:
                            case['status'] = 'spilled_or_unknown_resource_not_timed'
                            save()
                            continue
                        if not case['accuracy']['qualified'] and not args.time_unqualified:
                            case['status'] = 'accuracy_failed_not_timed'
                            save()
                            continue
                        for _ in range(3):
                            for function in (reference, call):
                                value = function()
                                del value
                        samples = {'reference': [], 'candidate': []}
                        for _ in range(args.rounds):
                            for label, function in [('reference', reference), ('candidate', call),
                                                    ('candidate', call), ('reference', reference)]:
                                samples[label].extend(_sample(function) for _ in range(args.samples))
                        case['summary'] = {label: {key: statistics.median(row[key] for row in values)
                                                   for key in ['cuda_ms', 'wall_ms']}
                                           for label, values in samples.items()}
                        case['samples'] = samples
                        case['memory'] = {'reference': _memory(reference), 'candidate': _memory(call)}
                        case['rng_unchanged'] = bool(torch.equal(before_rng, torch.cuda.get_rng_state()))
                        case['status'] = ('qualified_micro_only' if case['accuracy']['qualified']
                                          else 'accuracy_failed_research_timing_only')
                    except Exception as error:
                        case['status'] = 'compile_execution_or_resource_failure'
                        case['error'] = type(error).__name__ + ': ' + str(error)
                        save()
                        if type(error).__name__ not in ('OutOfResources', 'CompilationError',
                                                        'UnsupportedLanguageConstruct', 'PTXASError'):
                            raise
                    print(json.dumps({key: value for key, value in case.items() if key != 'samples'}), flush=True)
                    save()
                del expected, a
        del packed, state
    report['completed'] = True
    save()
    print('SM75_NF4_MICRO_COMPLETE', str(output_path), flush=True)


if __name__ == '__main__':
    main()

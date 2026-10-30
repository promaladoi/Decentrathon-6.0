"""Balanced native frozen-head diagnostic; no claim about training scores."""

import gc
import hashlib
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


def _summary(values):
    ordered = sorted(values)
    def percentile(p):
        point = (len(ordered) - 1) * p
        low = int(point)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (point - low)
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'q25_ms': percentile(0.25), 'q75_ms': percentile(0.75),
            'minimum_ms': min(values), 'maximum_ms': max(values)}


def _parity(actual, expected):
    integer = {torch.float16: torch.int16, torch.float32: torch.int32}
    same_dtype = actual.dtype == expected.dtype
    exact = same_dtype and bool(torch.equal(actual.contiguous().view(integer[actual.dtype]),
                                            expected.contiguous().view(integer[expected.dtype])))
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype),
            'reference_dtype': str(expected.dtype), 'bitwise_equal': exact,
            'values_equal': bool(torch.equal(actual, expected))}


def _samples(call, repeats):
    cuda_values, wall_values = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        before = time.perf_counter()
        begin.record()
        gradient = call()
        end.record()
        end.synchronize()
        wall_values.append(1000 * (time.perf_counter() - before))
        cuda_values.append(begin.elapsed_time(end))
        del gradient
    return {'cuda_event': _summary(cuda_values), 'wall': _summary(wall_values),
            'cuda_event_values_ms': cuda_values, 'wall_values_ms': wall_values}


def _memory(call):
    gc.collect()
    torch.cuda.synchronize()
    allocated_start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gradient = call()
    torch.cuda.synchronize()
    allocated_peak = torch.cuda.max_memory_allocated()
    allocated_end = torch.cuda.memory_allocated()
    del gradient
    torch.cuda.synchronize()
    return {'allocated_start_bytes': allocated_start, 'allocated_peak_bytes': allocated_peak,
            'extra_allocated_peak_bytes': allocated_peak - allocated_start,
            'allocated_end_with_result_bytes': allocated_end,
            'allocated_after_result_release_bytes': torch.cuda.memory_allocated()}


def run_tuning(rounds=3, samples=4, warmups=3):
    """Same-input CE+head input-gradient timing, balanced AB/BA in each round."""
    if rounds < 3 or samples < 1 or warmups < 1:
        raise ValueError('use >=3 rounds, >=1 sample and >=1 warmup')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Kaggle T4 required; no diagnostic was run')
    from .vocab import _half_cross_entropy
    from .frozen_head import _FrozenHead, _supported
    started = time.perf_counter()
    report = {'kind': 'synthetic_frozen_head_recompute_not_training_score',
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'rounds': rounds, 'samples_per_block': samples, 'warmups_per_variant': warmups,
              'seed': 319, 'ordering': 'each round: native,recompute then recompute,native',
              'timing': 'whole head+half CE+input gradient; CUDA events and synchronized wall',
              'memory': 'extra allocated PyTorch bytes for one call, not reserved or NVML peak',
              'candidate_includes_supported_guard': True,
              'weights': 'seeded random diagnostic frozen fp32 Linear; actual head dimensions',
              'training_score_claim': False, 'sources': {}, 'cases': []}
    for filename in ('frozen_head.py', 'benchmark_frozen_head.py', 'vocab.py'):
        report['sources'][filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    torch.manual_seed(319)
    layer = torch.nn.Linear(1536, 151936, bias=False, device='cuda').requires_grad_(False)
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.02)
    weight_version = layer.weight._version
    for rows in (256, 2048):
        x = torch.randn((rows, 1536), device='cuda', requires_grad=True)
        labels = torch.randint(151936, (rows,), device='cuda')
        labels[::4] = -100
        before_x, before_labels = x.detach().clone(), labels.clone()

        def evaluate(candidate, keep_outputs=False):
            with torch.autocast('cuda', dtype=torch.float16):
                logits = (_FrozenHead.apply(x, layer.weight) if candidate and _supported(layer, x)
                          else F.linear(x, layer.weight))
                loss = _half_cross_entropy(logits, labels)
            gradient, = torch.autograd.grad(loss, (x,))
            return (logits, loss, gradient) if keep_outputs else gradient

        with torch.autocast('cuda', dtype=torch.float16):
            eligible = bool(_supported(layer, x))
        case = {'rows': rows, 'vocab': 151936, 'hidden': 1536, 'input_dtype': str(x.dtype),
                'weight_dtype': str(layer.weight.dtype), 'eligible': eligible,
                'rounds': [], 'memory': {}}
        if not eligible:
            case['status'] = 'ineligible_no_candidate_measurement'
            report['cases'].append(case)
            del x, labels, before_x, before_labels
            continue
        expected = evaluate(False, True)
        actual = evaluate(True, True)
        case['parity'] = {name: _parity(a, b) for name, a, b in
                          zip(('logits', 'loss', 'input_gradient'), actual, expected)}
        case['all_bitwise_equal'] = all(item['bitwise_equal'] for item in case['parity'].values())
        del actual, expected
        if not case['all_bitwise_equal']:
            case['status'] = 'numerical_failure_no_timing'
            report['cases'].append(case)
            del x, labels, before_x, before_labels
            continue
        calls = {'native': lambda: evaluate(False), 'recompute': lambda: evaluate(True)}
        for _ in range(warmups):
            for name in ('native', 'recompute'):
                gradient = calls[name]()
                del gradient
        torch.cuda.synchronize()
        aggregate = {name: {'cuda_event': [], 'wall': []} for name in calls}
        for round_index in range(rounds):
            row = {'round': round_index + 1, 'blocks': []}
            for order in (('native', 'recompute'), ('recompute', 'native')):
                block = {'order': list(order), 'measurements': {}}
                for name in order:
                    measured = _samples(calls[name], samples)
                    block['measurements'][name] = measured
                    aggregate[name]['cuda_event'].extend(measured['cuda_event_values_ms'])
                    aggregate[name]['wall'].extend(measured['wall_values_ms'])
                row['blocks'].append(block)
            for metric in ('cuda_event', 'wall'):
                row[f'{metric}_speedup'] = statistics.median([
                    block['measurements']['native'][metric]['median_ms']
                    / block['measurements']['recompute'][metric]['median_ms'] for block in row['blocks']])
            case['rounds'].append(row)
        case['aggregate'] = {name: {metric: _summary(values) for metric, values in metrics.items()}
                             for name, metrics in aggregate.items()}
        for name in ('native', 'recompute'):
            case['memory'][name] = _memory(calls[name])
        case['inputs_unchanged'] = bool(torch.equal(x, before_x))
        case['labels_unchanged'] = bool(torch.equal(labels, before_labels))
        case['weight_version_unchanged'] = layer.weight._version == weight_version
        case['status'] = 'qualified_diagnostic_only'
        report['cases'].append(case)
        del x, labels, before_x, before_labels, calls
    report['status'] = ('completed_diagnostic_only' if all(
        case['status'] == 'qualified_diagnostic_only' for case in report['cases']) else 'qualification_failure')
    report['diagnostic_wall_seconds'] = time.perf_counter() - started
    return report

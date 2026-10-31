"""Standalone balanced Q/V-dropout diagnostic; not a training score.

The dense reference pads once and reuses that input for two dropout calls,
matching _Batch's cache. It releases padded values before backward, matching
done_with(x). Candidate production code never changes global RNG state; only
the untimed exactness comparison restores an identical initial generator state.
"""

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
            'q25_ms': percentile(.25), 'q75_ms': percentile(.75),
            'minimum_ms': min(values), 'maximum_ms': max(values)}


def _equal(actual, expected):
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    return {'dtype': str(actual.dtype), 'reference_dtype': str(expected.dtype),
            'shape': list(actual.shape),
            'bitwise_equal': bool(torch.equal(actual.view(torch.int32), expected.view(torch.int32))),
            'finite_bits_equal': bool(torch.equal(actual[finite].view(torch.int32),
                                                 expected[finite].view(torch.int32))),
            'nan_locations_equal': bool(torch.equal(torch.isnan(actual), torch.isnan(expected)))}


def _samples(call, count):
    device, wall = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(count):
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        result = call()
        end.record()
        end.synchronize()
        wall.append(1000 * (time.perf_counter() - started))
        device.append(begin.elapsed_time(end))
        del result
    return {'cuda_event': _summary(device), 'wall': _summary(wall),
            'cuda_event_values_ms': device, 'wall_values_ms': wall}


def _memory(call):
    gc.collect()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    end = torch.cuda.memory_allocated()
    del result
    torch.cuda.synchronize()
    return {'allocated_start_bytes': start, 'allocated_peak_bytes': peak,
            'extra_allocated_peak_bytes': peak - start,
            'allocated_with_result_bytes': end,
            'allocated_after_result_release_bytes': torch.cuda.memory_allocated()}


def run_tuning(rounds=3, samples=7, warmups=5):
    if rounds < 3 or samples < 1 or warmups < 1:
        raise ValueError('use >=3 rounds, >=1 sample and >=1 warmup')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Kaggle T4 required; no diagnostic was run')
    import triton
    from .dropout import real_row_dropout, _supported

    started = time.perf_counter()
    report = {'kind': 'paired_native_rng_dropout_diagnostic_not_training_score',
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'triton': triton.__version__, 'rounds': rounds, 'samples_per_block': samples,
              'warmups_per_variant': warmups, 'seed': 613, 'p': .05,
              'ordering': 'each round: reference,candidate then candidate,reference',
              'operation': 'two Q/V dropouts sharing one input, plus joined input gradient',
              'baseline_lifetime': 'one padded input shared by Q/V; released before backward',
              'measurement_rng': 'native generator advances normally; no state reset inside timed calls',
              'memory': 'extra allocated PyTorch peak of one pair; not NVML or full training peak',
              'candidate_installed_in_unpad': False, 'training_score_claim': False,
              'sources': {}, 'cases': []}
    for name in ('dropout.py', 'benchmark_dropout.py'):
        report['sources'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    torch.manual_seed(613)
    specifications = [(4, 128, 1536, (128, 96, 64, 32)),
                      (4, 256, 1536, (256, 192, 128, 96)),
                      (4, 384, 1536, (384, 192, 128, 64)),
                      (4, 512, 1536, (512, 250, 100, 22)),
                      (4, 512, 1536, (512, 512, 512, 512))]
    for batch, sequence, hidden, lengths in specifications:
        mask = torch.arange(sequence, device='cuda')[None] < torch.tensor(lengths, device='cuda')[:, None]
        rows = mask.reshape(-1).nonzero().squeeze(1)
        x = torch.randn((rows.numel(), hidden), device='cuda', dtype=torch.float32,
                        requires_grad=True)
        gq, gv = torch.randn_like(x) * 256., torch.randn_like(x) * 256.
        before_x, before_gq, before_gv = x.detach().clone(), gq.clone(), gv.clone()

        def evaluate(candidate, outputs=False):
            if candidate:
                q = real_row_dropout(x, rows, batch, sequence, .05, rows_from_nonzero=True)
                v = real_row_dropout(x, rows, batch, sequence, .05, rows_from_nonzero=True)
            else:
                padded = x.new_zeros((batch * sequence, hidden)).index_copy(0, rows, x)
                padded = padded.view(batch, sequence, hidden)
                q = F.dropout(padded, p=.05, training=True).view(-1, hidden).index_select(0, rows)
                v = F.dropout(padded, p=.05, training=True).view(-1, hidden).index_select(0, rows)
                del padded
            dx, = torch.autograd.grad((q, v), (x,), (gq, gv))
            return (q, v, dx) if outputs else dx

        initial = torch.cuda.get_rng_state()
        expected = evaluate(False, True)
        expected_after = torch.cuda.get_rng_state()
        expected_next = torch.rand(41, device='cuda')
        torch.cuda.set_rng_state(initial)
        actual = evaluate(True, True)
        actual_after = torch.cuda.get_rng_state()
        actual_next = torch.rand(41, device='cuda')
        parity = {name: _equal(a, b) for name, a, b in
                  zip(('q_output', 'v_output', 'input_gradient'), actual, expected)}
        case = {'shape': [batch, sequence, hidden], 'lengths': list(lengths),
                'real_rows': rows.numel(), 'real_fraction': rows.numel() / (batch * sequence),
                'eligible': _supported(x, rows, batch, sequence, .05, True, True),
                'parity': parity,
                'rng_after_equal': bool(torch.equal(actual_after, expected_after)),
                'following_random_draw_equal': bool(torch.equal(actual_next, expected_next)),
                'rounds': [], 'memory': {}}
        case['all_bitwise_equal'] = (all(v['bitwise_equal'] for v in parity.values())
                                     and case['rng_after_equal'] and case['following_random_draw_equal'])
        del actual, expected, expected_next, actual_next, initial, expected_after, actual_after
        if not case['all_bitwise_equal'] or not case['eligible']:
            case['status'] = 'numerical_failure_no_timing'
            report['cases'].append(case)
            del x, rows, mask, gq, gv, before_x, before_gq, before_gv
            continue
        calls = {'reference': lambda: evaluate(False), 'candidate': lambda: evaluate(True)}
        for _ in range(warmups):
            for name in ('reference', 'candidate'):
                result = calls[name]()
                del result
        torch.cuda.synchronize()
        aggregate = {name: {'cuda_event': [], 'wall': []} for name in calls}
        for round_index in range(rounds):
            row = {'round': round_index + 1, 'blocks': []}
            for order in (('reference', 'candidate'), ('candidate', 'reference')):
                block = {'order': list(order), 'measurements': {}}
                for name in order:
                    measured = _samples(calls[name], samples)
                    block['measurements'][name] = measured
                    for metric in ('cuda_event', 'wall'):
                        aggregate[name][metric].extend(measured[metric + '_values_ms'])
                row['blocks'].append(block)
            for metric in ('cuda_event', 'wall'):
                row[metric + '_speedup'] = statistics.median([
                    block['measurements']['reference'][metric]['median_ms']
                    / block['measurements']['candidate'][metric]['median_ms'] for block in row['blocks']])
            case['rounds'].append(row)
        case['aggregate'] = {name: {metric: _summary(values) for metric, values in metrics.items()}
                             for name, metrics in aggregate.items()}
        for name in ('reference', 'candidate'):
            case['memory'][name] = _memory(calls[name])
        case['inputs_and_gradients_unchanged'] = (bool(torch.equal(x, before_x))
                                                 and bool(torch.equal(gq, before_gq))
                                                 and bool(torch.equal(gv, before_gv)))
        case['saved_mask_bytes_estimate'] = {'reference_two_full_bool_masks': 2 * batch * sequence * hidden,
                                             'candidate_two_real_bool_masks': 2 * rows.numel() * hidden}
        case['status'] = 'qualified_diagnostic_only'
        report['cases'].append(case)
        del x, rows, mask, gq, gv, before_x, before_gq, before_gv, calls
    report['status'] = ('completed_diagnostic_only' if all(
        case['status'] == 'qualified_diagnostic_only' for case in report['cases']) else 'qualification_failure')
    report['diagnostic_wall_seconds'] = time.perf_counter() - started
    return report

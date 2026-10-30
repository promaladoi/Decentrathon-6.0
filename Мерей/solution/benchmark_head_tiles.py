"""Standalone same-math head chunk benchmark; never imported by solution.apply.

Run on an idle T4, in the pinned environment. This diagnostic varies only
internal work tiling, preserving all rows, statistics and supervised gradients.
Its tensor allocation peaks are not whole-training memory qualification.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from solution import lmhead


def timed(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return statistics.median(measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--chunks', type=int, nargs='+', default=[128, 256, 384, 512])
    args = parser.parse_args()
    assert 'T4' in torch.cuda.get_device_name()
    torch.manual_seed(20261030)
    device, vocab, hidden = 'cuda', 151936, 1536
    weight = torch.randn((vocab, hidden), device=device, dtype=torch.float16).mul_(0.02)
    cases = []
    for rows, supervised in [(512, 320), (832, 528), (1280, 896), (2048, 1344)]:
        x = torch.randn((rows, hidden), device=device, dtype=torch.float16)
        order = torch.randperm(rows, device=device)
        labelled, prompt = order[:supervised], order[supervised:]
        targets = torch.full((rows,), -100, device=device, dtype=torch.int64)
        targets[labelled] = torch.randint(vocab, (supervised,), device=device)
        entropy = torch.empty(rows, device=device, dtype=torch.float32)
        argmax = torch.empty(rows, device=device, dtype=torch.int64)
        ce = torch.empty(rows, device=device, dtype=torch.float32)
        dx = torch.empty_like(x)
        scale = torch.tensor(1024.0 / supervised, device=device)

        def operation():
            saved_lse = []
            lmhead._stats(x, weight, labelled, targets, entropy, argmax, ce,
                          precompute=(scale, dx, saved_lse))
            lmhead._stats(x, weight, prompt, targets, entropy, argmax, None)

        lmhead.CHUNK = 512
        operation()
        reference = {'entropy': entropy.clone(), 'argmax': argmax.clone(),
                     'ce': ce[labelled].clone(), 'dx': dx[labelled].clone()}
        trials = []
        # Reverse second sweep reduces simple drift bias. Each value remains
        # an internal tiling choice; no training data or config is altered.
        for chunk in [*args.chunks, *reversed(args.chunks)]:
            lmhead.CHUNK = chunk
            began = time.perf_counter()
            operation()
            torch.cuda.synchronize()
            first_s = time.perf_counter() - began
            actual = {'entropy': entropy, 'argmax': argmax, 'ce': ce[labelled],
                      'dx': dx[labelled]}
            errors = {}
            for name, value in actual.items():
                expected = reference[name]
                if name == 'argmax':
                    errors[name + '_different'] = int((value != expected).sum())
                else:
                    gap = (value.float() - expected.float()).abs()
                    errors[name + '_max_abs'] = float(gap.max())
                    errors[name + '_relative_l2'] = float(gap.norm() / expected.float().norm().clamp_min(1e-30))
                    assert torch.isfinite(value).all()
            assert errors['entropy_max_abs'] < 0.01 and errors['ce_max_abs'] < 0.01
            assert errors['dx_relative_l2'] < 0.003
            trial = {'chunk': chunk, 'median_gpu_ms': timed(operation, args.repeats),
                     'first_call_s': first_s, **errors}
            trials.append(trial)
            print('HEAD_TRIAL', json.dumps({'rows': rows, 'supervised': supervised, **trial}), flush=True)
        summaries = {str(chunk): statistics.median(t['median_gpu_ms'] for t in trials if t['chunk'] == chunk)
                     for chunk in args.chunks}
        cases.append({'rows': rows, 'supervised': supervised, 'median_gpu_ms': summaries, 'trials': trials})
        print('HEAD_CASE', json.dumps(cases[-1]), flush=True)
    lmhead.CHUNK = 512
    report = {'kind': 'head_chunk_diagnostic_not_training_admission', 'cases': cases,
              'torch_max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'torch_max_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
    args.out.write_text(json.dumps(report, indent=2))
    print('HEAD_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

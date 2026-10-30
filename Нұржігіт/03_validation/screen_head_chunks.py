"""Synthetic private-head chunk screen; not a training result or quality gate."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import solution.lmhead as head
    assert 'T4' in torch.cuda.get_device_name(0)
    torch.manual_seed(4602)
    weight = torch.randn((151936, 1536), device='cuda', dtype=torch.float16) * .02
    original_chunk, original_scale = head.CHUNK, head._private_expected_loss_scale
    head._private_expected_loss_scale = lambda: 1024.0
    report = {'kind': 'synthetic head forward/backward screen, not full-task admission', 'cases': []}
    try:
        for rows in (512, 1024, 1536):
            x = torch.randn((rows, 1536), device='cuda', dtype=torch.float32, requires_grad=True)
            targets = torch.randint(weight.shape[0], (rows,), device='cuda')
            targets[::3] = -100
            denominator = (targets != -100).sum()

            def run(chunk):
                head.CHUNK = chunk
                stats = {}
                loss = head.LMHeadLoss.apply(x, weight, targets, denominator, stats)
                grad, = torch.autograd.grad(loss, x, grad_outputs=torch.full_like(loss, 1024.0))
                return loss.detach(), grad, stats['entropy'], stats['argmax']

            reference = run(512)
            samples = {chunk: [] for chunk in (128, 256, 512)}
            peaks = {chunk: [] for chunk in samples}
            errors = {}
            for chunk in samples:
                result = run(chunk)
                for actual, expected in zip(result[:3], reference[:3]):
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-5)
                # Record argmax ties/differences rather than hide them.
                errors[chunk] = {'loss_abs': abs(result[0].item() - reference[0].item()),
                                 'gradient_max_abs': (result[1] - reference[1]).abs().max().item(),
                                 'argmax_differences': (result[3] != reference[3]).sum().item()}
                del result
            for repeat in range(7):
                order = (128, 256, 512) if repeat % 2 else (512, 256, 128)
                for chunk in order:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_bytes = torch.cuda.memory_allocated()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    result = run(chunk)
                    end.record()
                    end.synchronize()
                    samples[chunk].append(begin.elapsed_time(end))
                    peaks[chunk].append((torch.cuda.max_memory_allocated() - baseline_bytes) / 2**20)
                    del result
            case = {'rows': rows, 'median_ms': {k: statistics.median(v) for k, v in samples.items()},
                    'extra_peak_mib': {k: max(v) for k, v in peaks.items()}, 'errors': errors}
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
            del reference, x, targets, denominator
    finally:
        head.CHUNK, head._private_expected_loss_scale = original_chunk, original_scale
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

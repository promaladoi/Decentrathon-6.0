"""v50: isolated native BMM screen, never a full-task score or active patch.

Compare separate gate/up FP16 GEMMs with a batch of two. Each projection
and each input gradient must retain its original FP16 rounding. The FP32
sum of those gradients is also compared. Prepacked weights/gradients give
an optimistic timing bound; stack-inclusive timing includes layout costs.
No model, training settings, harness, or selected solution is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep the prior screen result')
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if processes.strip():
        raise RuntimeError('Run this isolated screen only while both GPUs are idle')
    import torch
    import torch.nn.functional as F
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_capability(0) == (7, 5), 'A T4 is required'
    torch.cuda.set_device(0)
    report = dict(version='v50', kind='isolated inactive native-BMM research',
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  production_solution_changed=False, full_task_admission='not run',
                  target_total_seconds=27, target_claimed=False, rows=[])

    def compare(expected, actual):
        diff = (expected.float() - actual.float()).abs()
        return dict(exact=bool(torch.equal(expected, actual)),
                    mismatches=int((expected != actual).sum().item()),
                    elements=expected.numel(), max_abs=float(diff.max().item()))

    def milliseconds(function, repeats=20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats

    for tokens in (32, 256, 512, 832, 1024, 1536, 2048):
        for seed in (1, 42):
            torch.manual_seed(seed)
            x = torch.randn(tokens, 1536, device='cuda', dtype=torch.float16) * .1
            weights = torch.randn(2, 8960, 1536, device='cuda', dtype=torch.float16) * .03
            gradients = torch.randn(2, tokens, 8960, device='cuda', dtype=torch.float16) * .02
            wg, wu = weights.unbind()
            dg, du = gradients.unbind()
            def separate_forward():
                return F.linear(x, wg), F.linear(x, wu)
            def batched_forward():
                return torch.bmm(x.unsqueeze(0).expand(2, -1, -1), weights.transpose(1, 2))
            def separate_backward():
                return dg @ wg, du @ wu
            def batched_backward():
                return torch.bmm(gradients, weights)
            def stacked_forward():
                return torch.bmm(x.unsqueeze(0).expand(2, -1, -1),
                                 torch.stack((wg, wu)).transpose(1, 2))
            def stacked_backward():
                return torch.bmm(torch.stack((dg, du)), torch.stack((wg, wu)))
            f0 = torch.stack(separate_forward())
            f1 = batched_forward()
            b0 = torch.stack(separate_backward())
            b1 = batched_backward()
            row = dict(tokens=tokens, seed=seed, forward=compare(f0, f1),
                       backward=compare(b0, b1),
                       gradient_sum=compare(b0[0].float() + b0[1].float(),
                                            b1[0].float() + b1[1].float()))
            exact = all(row[k]['exact'] for k in ('forward', 'backward', 'gradient_sum'))
            row['exact'] = exact
            # Timing does not override the numerical gate. Reject inexact cases.
            if exact:
                operations = dict(separate_forward=separate_forward, batched_forward=batched_forward,
                                  separate_backward=separate_backward, batched_backward=batched_backward,
                                  stacked_forward=stacked_forward, stacked_backward=stacked_backward)
                for fn in operations.values():
                    for _ in range(5):
                        fn()
                samples = {key: [] for key in operations}
                for repeat in range(6):
                    keys = list(operations)
                    if repeat % 2:
                        keys.reverse()
                    for key in keys:
                        samples[key].append(milliseconds(operations[key]))
                row['median_ms'] = {key: statistics.median(v) for key, v in samples.items()}
                row['samples_ms'] = samples
            report['rows'].append(row)
            print('BATCHED_MLP', json.dumps(row), flush=True)
            del x, weights, gradients, wg, wu, dg, du, f0, f1, b0, b1
    all_exact = all(row['exact'] for row in report['rows'])
    report['all_exact'] = all_exact
    report['status'] = 'screen_complete' if all_exact else 'rejected_numerical_gate'
    report['note'] = ('Even a passing screen needs native NF4 dispatch, full MLP tests, '
                      'original paired full-task evaluation and all ten original checks.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('BATCHED_MLP_OUTCOME', json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
    return 0 if all_exact else 2


if __name__ == '__main__':
    raise SystemExit(main())

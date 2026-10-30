"""Exact decoder launch screen on T4; timings are not a full-task score.

Uses the existing team's kernel and native arithmetic, no package/model/cache
changes. Every launch configuration must match the current decoder bitwise.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import triton
    import bitsandbytes.functional as bnb
    from solution.nf4 import nf4_dequant_kernel, _decode
    assert 'T4' in torch.cuda.get_device_name(0)
    torch.manual_seed(4601)
    configurations = [(512, 4), (1024, 4), (2048, 4), (4096, 4),
                      (1024, 8), (2048, 8), (4096, 8)]
    report = {'kind': 'isolated exact decoder screen, not a contest score',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'triton': triton.__version__, 'cases': []}
    for shape in [(256, 1536), (1536, 1536), (8960, 1536), (1536, 8960)]:
        weight = torch.randn(shape, device='cuda', dtype=torch.float16)
        packed, state = bnb.quantize_4bit(weight, blocksize=64, quant_type='nf4', compress_statistics=True)
        reference = _decode(packed, state, torch.float16)
        output = torch.empty_like(reference)
        del weight
        n = output.numel()

        def launch(config):
            block, warps = config
            nf4_dequant_kernel[(triton.cdiv(n, block),)](
                packed, state.absmax, state.code, state.state2.absmax, state.state2.code,
                state.offset, output, n, 64, True, 256, block,
                num_warps=warps, enable_fp_fusion=False)

        for config in configurations:
            launch(config)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        samples = {str(config): [] for config in configurations}
        # Rotate/reverse order rather than giving one setting a hotter GPU.
        for repeat in range(9):
            order = configurations[repeat % len(configurations):] + configurations[:repeat % len(configurations)]
            if repeat % 2:
                order = order[::-1]
            for config in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    launch(config)
                end.record()
                end.synchronize()
                samples[str(config)].append(start.elapsed_time(end) / 30)
        medians = {key: statistics.median(values) for key, values in samples.items()}
        case = {'shape': shape, 'all_bitwise_equal': True, 'median_ms': medians, 'samples_ms': samples}
        report['cases'].append(case)
        print(json.dumps({key: value for key, value in case.items() if key != 'samples_ms'}), flush=True)
        del packed, state, reference, output
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

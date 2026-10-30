"""CPU-only ownership probe against the exact previous Git source.

This does not measure CUDA memory, kernel arithmetic or full-task time.
Run separately from any measured training process.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from solution.test_head_lifetime import _exercise_python_stats

BASE = '2d171ddc1f6d43e593f5dd80e1a86731f27ad6f7'


def old_stats(namespace):
    source = subprocess.check_output(['git', 'show', BASE + ':solution/lmhead.py'], cwd=REPO)
    tree = ast.parse(source)
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_stats']
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact v44 _stats>', 'exec'), namespace)
    return namespace['_stats']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {'kind': 'CPU Python ownership only; GPU kernels replaced by no-retention stubs',
              'baseline_commit': BASE, 'cases': {},
              'fixed_task_full_fp16_logit_chunk_mib': 512 * 151936 * 2 / 2**20,
              'full_task_gpu_peak_saving_measured': False, 'full_task_time_measured': False}
    for mode in ('prompt', 'private', 'saved'):
        before = _exercise_python_stats(mode, old_stats)
        after = _exercise_python_stats(mode)
        assert before[1] == after[1] == [3, 3, 3, 2]
        expected_before = [0, 1, 2, 3] if mode == 'saved' else [0, 1, 1, 1]
        expected_after = expected_before if mode == 'saved' else [0, 0, 0, 0]
        assert before[0] == expected_before and after[0] == expected_after
        report['cases'][mode] = {'v44_previous_chunks_alive_before_next_mm': before[0],
                                 'v45_previous_chunks_alive_before_next_mm': after[0],
                                 'gemm_row_schedule_unchanged': before[1] == after[1]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Compile/load-only MLP-pair resource and PTX diagnostic; never launch math.

python solution/diagnose_mlp_grad_pair.py --out /kaggle/working/mlp_pair_resources.json --dump-ir
Default compares the original M32N32K32 tile with the one row-scale revision.
It reads CompiledKernel.run to initialize handles, without calling the launcher.
This is outside training/score measurements and provides no performance claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.mlp_grad_pair import (CONFIGS, DEFAULT_CONFIG, ROW_SCALE_CONFIG,
                                   mlp_grad_pair_kernel, mlp_grad_pair_row_scale_kernel)


def _metadata(value):
    if hasattr(value, '_asdict'):
        return dict(value._asdict())
    if isinstance(value, dict):
        return value
    return vars(value) if hasattr(value, '__dict__') else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', nargs='+', default=[DEFAULT_CONFIG, ROW_SCALE_CONFIG])
    parser.add_argument('--rows', type=int, default=833)
    parser.add_argument('--k', type=int, default=8960)
    parser.add_argument('--n', type=int, default=1536)
    parser.add_argument('--dump-ir', action='store_true')
    args = parser.parse_args()
    configs = list(CONFIGS) if args.configs == ['all'] else args.configs
    if min(args.rows, args.k, args.n) <= 0 or any(c not in CONFIGS for c in configs):
        parser.error('positive dimensions and known tile names are required')
    if ROW_SCALE_CONFIG in configs and args.n % 64:
        parser.error('the row-scale revision requires N divisible by 64')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA compiler target/driver is required')
    device = torch.cuda.get_device_properties(0)
    report = {'kind': 'mlp_pair_compile_load_only_no_execution', 'kernel_math_executed': False,
              'torch': torch.__version__, 'triton': triton.__version__, 'device': device.name,
              'compute_capability': [device.major, device.minor],
              'shared_memory_per_block_bytes': device.shared_memory_per_block,
              'rows': args.rows, 'k': args.k, 'n': args.n,
              'source_sha256': hashlib.sha256((ROOT / 'solution/mlp_grad_pair.py').read_bytes()).hexdigest(),
              'ptx_counts_are_static_not_dynamic_instruction_counts': True, 'configurations': []}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    ir_dir = output.parent / (output.stem + '_ir')
    if args.dump_ir:
        ir_dir.mkdir(parents=True, exist_ok=True)
    f16, f32, u8 = torch.float16, torch.float32, torch.uint8
    pointer_types = (f16, f16,
                     u8, u8, f32, f32, f32, f32,
                     u8, u8, f32, f32, f32, f32, f32)
    for config in configs:
        bm, bn, bk, warps, stages = CONFIGS[config]
        kernel = mlp_grad_pair_row_scale_kernel if config == ROW_SCALE_CONFIG else mlp_grad_pair_kernel
        row = {'config': config, 'kernel': ('mlp_grad_pair_row_scale_kernel'
                                           if config == ROW_SCALE_CONFIG else 'mlp_grad_pair_kernel'),
               'tile': CONFIGS[config],
               'ctas': triton.cdiv(args.rows, bm) * triton.cdiv(args.n, bn),
               'reduction_iterations': triton.cdiv(args.k, bk),
               'nominal_four_half_operand_staging_bytes': 4 * bk * (bm + bn) * stages}
        report['configurations'].append(row)
        started = time.perf_counter()
        try:
            compiled = kernel.warmup(*pointer_types, args.rows, args.n, args.k,
                                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                     grid=(triton.cdiv(args.rows, bm), triton.cdiv(args.n, bn)),
                                     num_warps=warps, num_stages=stages, enable_fp_fusion=False)
            # Property access loads the CUDA module/launcher handles only.
            # No ``compiled.run(...)`` or JITKernel[...] execution occurs.
            getattr(compiled, 'run', None)
            row['compile_load_s'] = time.perf_counter() - started
            row['n_regs'] = getattr(compiled, 'n_regs', None)
            row['n_spills'] = getattr(compiled, 'n_spills', None)
            row['metadata'] = _metadata(getattr(compiled, 'metadata', None))
            ptx = compiled.asm.get('ptx', '')
            patterns = {'mma_sync': r'\bmma\.sync\b', 'ldmatrix': r'\bldmatrix\.',
                        'load_local': r'\bld\.local\b', 'store_local': r'\bst\.local\b',
                        'load_global': r'\bld\.global\b', 'barrier': r'\bbar\.sync\b',
                        'fma_f32': r'\bfma\.[^;\n]*\.f32\b',
                        'shared_decl': r'\.shared\b'}
            row['ptx_instruction_counts'] = {name: len(re.findall(pattern, ptx))
                                             for name, pattern in patterns.items()}
            row['ptx_chars'] = len(ptx)
            row['tensor_core_instructions_present'] = row['ptx_instruction_counts']['mma_sync'] > 0
            if args.dump_ir:
                paths = {}
                for kind in ('ptx', 'ttgir', 'ttir', 'llir'):
                    value = compiled.asm.get(kind)
                    if isinstance(value, str):
                        target = ir_dir / (config + '.' + kind)
                        target.write_text(value, encoding='utf-8')
                        paths[kind] = str(target)
                row['ir_paths'] = paths
            row['status'] = 'compiled_and_handles_loaded_without_execution'
        except Exception as error:
            row['compile_load_s'] = time.perf_counter() - started
            row['status'] = 'compile_or_handle_load_failed'
            row['error'] = type(error).__name__ + ': ' + str(error)
        output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(json.dumps(row, default=str), flush=True)
    print('MLP_PAIR_RESOURCE_DIAGNOSTIC_COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()

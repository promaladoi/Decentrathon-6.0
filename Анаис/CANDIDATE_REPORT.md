# Benchmark report

PASS: 10 official checks, matching result SHA256

Result SHA256: `b7d5b8303a117c936d993752731dd8b4a66eb5e10e9316e4e2015f4336e6d1b2`

| Run | Count | Time, s (median) | Peak MiB (max) | Eval loss |
|---|---:|---:|---:|---:|
| base | 1 | 81.26 | 14345 | 2.049650 |
| solution | 1 | 59.34 | 5641 | 2.049640 |

Speedup: 1.3694x; time saved: 26.975%.
Memory saved: 60.676%; eval loss change: -0.0005%.

Negative time savings mean a slowdown. Negative loss change means lower loss.
These public-data measurements do not establish hidden-data performance or final jury admission.
Checks authenticate result bytes, not the current solution files; rerun checks after code changes.

## Official checks
- PASS 1. every base and solution run finished: 1 base and 1 solution runs of 1 each
- PASS 2. no ready-made kernel libraries: loaded: none; imported in solution/: none; packages changed during the run: none
- PASS 3. own @triton.jit kernels in solution/ run in training: add_rmsnorm_bwd_kernel (solution/rmsnorm.py), add_rmsnorm_fwd_kernel (solution/rmsnorm.py), heads_rows_kernel (solution/unpad.py), lm_ce_kernel (solution/lmhead.py), nf4_dequant_kernel (solution/nf4.py), rmsnorm_bwd_kernel (solution/rmsnorm.py), rmsnorm_fwd_kernel (solution/rmsnorm.py), rope_pad_bwd_kernel (solution/unpad.py), rope_pad_fwd_kernel (solution/unpad.py); not launched in every solution run: _rotate_heads, rope_kernel; the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
- PASS 4. own kernels together take >= 5% of the training time: 12.6% of the training time (lowest of 1 runs, steps 3+)
- PASS 5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind): identical within tolerance
- PASS 6. loss curve follows the base: solution_1: first step 0.02%, mean of all steps 0.01% (max 1% and 3%)
- PASS 7. quality: eval loss at most 0.5% above the base: -0.000%
- PASS 8. not worse than the base on time or memory (10% margin): time x0.730, memory x0.393 of the base
- PASS 9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree): clean
- PASS 10. Soup regression tests with your patches: 79 passed, 14 warnings in 1.12s

## Training loss

| Run | Logged steps | First loss | Mean loss | First gap % | Mean relative gap % |
|---|---:|---:|---:|---:|---:|
| base_1 | 64 | 2.058933 | 2.056945 | 0.000000 | 0.000000 |
| solution_1 | 64 | 2.059355 | 2.057031 | 0.020496 | 0.012542 |

Gaps use base_1 as reference. Limits: first step 1%, mean relative gap 3%.
Training loss and held-out evaluation loss measure different data and must not be conflated.

## Individual timings

| Run | Wall s | Peak MiB | Exit code |
|---|---:|---:|---:|
| base_1 | 81.26 | 14345 | 0 |
| solution_1 | 59.34 | 5641 | 0 |

Two repeats describe variability but are insufficient for a statistical significance claim.

## Environment
```json
{
  "python": "3.13.15",
  "soup-cli": "0.77.0",
  "torch": "2.11.0+cu128",
  "triton": "3.6.0",
  "transformers": "5.16.1",
  "peft": "0.20.0",
  "trl": "0.29.1",
  "bitsandbytes": "0.50.2",
  "accelerate": "1.14.0",
  "datasets": "4.8.5",
  "gpus": [
    "Tesla T4, 580.178.04",
    "Tesla T4, 580.178.04"
  ]
}
```

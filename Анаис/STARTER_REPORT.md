# Benchmark report

NOT VERIFIED: official checks failed, absent or stale

Result SHA256: `264f8578bc90a07fcf94b51de8f11c538347ecf3d6aadc3c7add12aa25145e80`

| Run | Count | Time, s (median) | Peak MiB (max) | Eval loss |
|---|---:|---:|---:|---:|
| base | 2 | 84.34 | 14345 | 2.049660 |
| solution | 2 | 85.32 | 12385 | 2.049670 |

Speedup: 0.9886x; time saved: -1.156%.
Memory saved: 13.663%; eval loss change: +0.0005%.

Negative time savings mean a slowdown. Negative loss change means lower loss.
These public-data measurements do not establish hidden-data performance or final jury admission.
Checks authenticate result bytes, not the current solution files; rerun checks after code changes.

## Official checks
- PASS 1. every base and solution run finished: 2 base and 2 solution runs of 2 each
- PASS 2. no ready-made kernel libraries: loaded: none; imported in solution/: none; packages changed during the run: none
- FAIL 3. own @triton.jit kernels in solution/ run in training: none ran in every run; the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
- FAIL 4. own kernels together take >= 5% of the training time: 0.0% of the training time (lowest of 2 runs, steps 3+)
- PASS 5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind): identical within tolerance
- PASS 6. loss curve follows the base: base_2: first step 0.00%, mean of all steps 0.01%; solution_1: first step 0.00%, mean of all steps 0.01%; solution_2: first step 0.00%, mean of all steps 0.01% (max 1% and 3%)
- PASS 7. quality: eval loss at most 0.5% above the base: +0.000%
- PASS 8. not worse than the base on time or memory (10% margin): time x1.012, memory x0.863 of the base
- PASS 9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree): clean
- PASS 10. Soup regression tests with your patches: 79 passed, 14 warnings in 2.92s

## Training loss

| Run | Logged steps | First loss | Mean loss | First gap % | Mean relative gap % |
|---|---:|---:|---:|---:|---:|
| base_1 | 64 | 2.058933 | 2.056958 | 0.000000 | 0.000000 |
| base_2 | 64 | 2.058933 | 2.057003 | 0.000000 | 0.011001 |
| solution_1 | 64 | 2.058935 | 2.056994 | 0.000139 | 0.010184 |
| solution_2 | 64 | 2.058935 | 2.056971 | 0.000139 | 0.010354 |

Gaps use base_1 as reference. Limits: first step 1%, mean relative gap 3%.
Training loss and held-out evaluation loss measure different data and must not be conflated.

## Individual timings

| Run | Wall s | Peak MiB | Exit code |
|---|---:|---:|---:|
| base_1 | 83.92 | 14345 | 0 |
| base_2 | 84.77 | 14345 | 0 |
| solution_1 | 85.12 | 12385 | 0 |
| solution_2 | 85.52 | 12385 | 0 |

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

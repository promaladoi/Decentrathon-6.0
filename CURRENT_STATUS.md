# Current optimization status

- **Accepted console result: v13** (`13eaa8a`): 1.398x speedup, 5761 MiB,
  eval loss 2.04966, 106 kernel tests + 79 Soup tests, 10/10 checks.
  Final output supplied by participant; raw archive still awaits collection.
- **Fully archived fallback: v12** (`bd2b49a`): 1.1608x, 7385 MiB, 10/10,
  source and report hashes verified locally.
- **Rejected v14** (`fd5e482`): 4 exact dropout tests failed, 106 passed.
  RNG state matched but masks did not. Benchmark stopped before training.
  Raw supplied log is under `03_validation/reports/codex_v14_failed/`.
- **v14 repair** (`3016d55`): uses the original functional.dropout operation
  to produce a boolean mask; includes 6 exact-value/gradient/RNG-sequence tests.
  Pending GPU validation. No speed or quality claim.
- **v15 experiment** (`ef10dd1` computation): includes the mask repair and
  compiles the actual parallel-head/dropout kernels while the model loads.
  Pending 113 kernel tests, 79 Soup tests, two paired runs and all 10 checks.
  Syntax/whitespace validation passed locally; local Python has no CUDA/Triton.
- `04_submission/run_candidate.py` checks untouched task files and an idle T4,
  then runs dropout tests before the full suite and benchmark. A failed test
  stops training and still produces a source/log archive. No fake result JSON.
- Browser automation remains disconnected/blocked. The participant started v14
  manually and supplied its failing test output. No autonomous v15 run is claimed.
- Best-known team peer summary is Vladimir 1.468x / 5641 MiB (user supplied).
  No 70–90% reduction, victory or hidden-data quality guarantee is claimed.

# Twenty-minute optimization: measured final decision

Selected **v69**, exact measured source `6d91d6430f88b5786ba905fc931ff7a9de164c2d`. **20 seconds AND 500 MiB target: NOT MET.**

| Metric | Historical admitted v69 | Final admitted v69 |
|---|---:|---:|
| Whole command | 46.73 s | 46.73 s |
| NVML peak | 3959 MiB | 3959 MiB |
| Original correctness | 10/10 | 10/10 |
| Eval loss | 2.04962 | 2.04962 |

Current exact v69 solution-only control: **43.20 s / 3959 MiB**, two runs. This is a timing control, not a new ten-check admission.

| Candidate | Whole command | Peak MiB | Original checks | Nonincreasing paired reported loss |
|---|---:|---:|---:|---|
| v69 | 46.73 s | 3959 | 10/10 | True |
| v70 | 48.30 s | 3959 | 10/10 | False |
| v71 | 49.06 s | 3959 | 10/10 | True |
| v72 | 48.61 s | 3959 | 10/10 | True |

The original benchmark measures the entire child `soup train` command and the maximum sum of `nvidia-smi memory.used` over all GPUs, sampled every 100 ms. It does not score only kernel buffers or PyTorch allocated memory.

The actual first-forward diagnosis found a native Rich refresh thread blocking the direct-dropout guard. v72 moves native dashboard rendering into its main-thread update callback, preserving all state fields, outputs and strict foreign-thread fallback. The new Philox gather itself reproduces native full-padded RNG coordinates and counter progress. v70/v71 and failed timing-screen evidence are preserved, rather than presented as speed wins.

Memory evidence: registered CUDA parameter storage alone was **1078.74 MiB**. NF4 matrices plus the exact FP16 embedding have a theoretical payload lower bound of **1069.875 MiB**, excluding metadata, gradients, activations, workspaces and context. Even hypothetical four-bit matrices and embedding require **736.031 MiB**. The mandatory pre-solution context measured **107 MiB**. Therefore 500 MiB cannot hold the complete resident weight payload. This is not a proof against every CPU streaming algorithm; no <=20-second / <=500-MiB offloading implementation was validated in this window.

Reproduce from the selected exact source on one otherwise idle T4, with the pinned project dependencies and prepared data:

```bash
python harness/run_task.py --repeats 2 --out runs_final_recheck
python harness/check_rules.py --result runs_final_recheck/result.json
python harness/score.py runs_final_recheck/result.json
```

Full tests plus the same original benchmark: `python 04_submission/run_candidate.py --version v69_recheck`. [Pinned Kaggle cell](04_submission/kaggle_v69_cell.py).

Changed candidate files: `solution/direct_dropout.py`, `solution/lora_remat.py`, `solution/synchronous_display.py`, `solution/__init__.py` and their qualification tests. Only the selected exact source is installed in `solution/`; other candidate code remains in `experiments/`. All 15 protected organizer files are byte-identical. [Selection/source proof](04_submission/twenty_minute_selection.json).

The direct kernel was active in both v72 training runs: 1457 launches including compile warmup; timed GPU work after step 2 was 89.791 ms and 82.000 ms. This isolated work reduction did not improve whole-command latency.

[All nine original evidence files, verified by SHA-256](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-v69-v70-20261031).

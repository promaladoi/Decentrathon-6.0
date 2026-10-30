# Current status

All rows below are original paired T4 reports with two base and two solution
runs, all ten checks, and source/result hashes independently verified locally.

| Version | Exact code | Base / solution seconds | Peak MiB | Eval loss delta | Checks |
|---|---|---:|---:|---:|---:|
| v16 | `4a5cfd2` | 85.07 / 60.41 | 5601 | +0.001% | 10/10 |
| v17r1 | `7c5f6d8` | 84.28 / 54.19 | 5601 | -0.002% | 10/10 |
| v18r1 | `d01b2d8` | 80.94 / 55.61 | 4701 | +0.000% | 10/10 |
| v20r1 | `22ff279` | 81.23 / 55.81 | 4467 | -0.000% | 10/10 |
| v21 | `a67816b` | 80.84 / 55.48 | 4467 | -0.001% | 10/10 |
| v22 | `7a8b38f` | 80.70 / 54.64 | 4467 | +0.001% | 10/10 |
| v23 | `d8ac022` | 80.68 / 52.76 | 6747 | -0.003% | 10/10 |

v23 is the shortest collected absolute wall time: 34.6% below its paired base,
1.5292x speedup, 53.0% lower peak memory, own real kernels 7.9% of timed training.
It spends more VRAM on exact process-local frozen-weight reuse. v17r1 has the
highest observed paired ratio (1.5553x), but its first base was 87.03 s compared
with 81.52 s for its second. Do not rank close versions from unlike cold baselines
alone. v20r1/v21/v22 retain the lowest collected peak, 4467 MiB.

v24 (`053c9c4`) passed 214 solution tests, 79 Soup tests and all ten original
checks in the Kaggle console: base 77.09/80.89 s, solution 50.37/51.52 s,
peak 6651 MiB, eval delta -0.001%. Raw local collection is queued.
It uses native PyTorch variable-length causal attention, preserving per-example
boundaries, positions, full-shape dropout draws and exact task settings.

v25 (`fd296f4`) passed 230 solution tests, 79 Soup tests and all ten original
checks in the Kaggle console: base 80.65/81.09 s, solution 48.36/49.57 s,
peak 6651 MiB, eval delta +0.000%, real own-kernel share 8.1%. Collection pending.
v26 (`79643de`) passed 233 solution tests and 79 Soup tests. Its two training
commands took 45.86/46.91 s; eval and final admission checks are still running.
v27 (`7efba21`) and v28 (`4aaaade`) are queued afterward. No current user action
is required. v29 (`9b0d96b`) adds an inactive two-product MLP-gradient fusion for
a separate native-vs-own T4 diagnostic after the paired jobs finish.

A separate diagnostic of v17r1 completed: GPU attention forward/backward was
about 14.9% of recorded GPU kernel time; NF4 decoding about 9.2%. CPU profiling
identified repeated package metadata scans and dataset fingerprint serialization
as startup candidates. Profiler overhead is material: diagnostic wall time is
never a contest score, and these percentages are not full-command speedups.

The user target remains 20-30 seconds for the full command; it is not achieved.
The official harness, task and organizer tests remain unchanged. Browser control
works and no user action is needed to launch the current experiment. Initial
dependency failures and all later evidence are preserved separately.

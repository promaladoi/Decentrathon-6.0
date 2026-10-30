# v45 candidate pending T4

Head lifetime and own storage-kernel warm-up candidate.
8 targeted CPU checks passed; 5 CUDA checks skipped. No new GPU result.
Running v44 remains pinned to its own source and queue.

# v44 candidate pending T4

New combined memory candidate: see CODEX_V44_EXPERIMENT.md.
CPU checks passed; no new GPU result or promotion is claimed.

# Verified candidate comparison

Shortest collected median total time: **v34, 78.68 → 43.91 seconds**,
**1.7918x**, **44.2% less elapsed time**, **10/10**.
Peak **7151 MiB** (50.1% less than its base), eval delta
**-0.001%**. Exact measured code: `fe2272c79bf00ae17654c388c5e79237d265f968`.
Lower-memory alternative: **v35, 46.46 seconds,
4651 MiB**, all ten checks.

| Version | Code | Base s | Solution s | Peak MiB | Speedup | Eval delta | Checks |
|---|---|---:|---:|---:|---:|---:|---:|
| v16 | `4a5cfd2` | 85.07 | 60.41 | 5601 | 1.4082x | +0.001% | 10/10 |
| v17r1 | `7c5f6d8` | 84.28 | 54.19 | 5601 | 1.5553x | -0.002% | 10/10 |
| v18r1 | `d01b2d8` | 80.94 | 55.61 | 4701 | 1.4555x | +0.000% | 10/10 |
| v20r1 | `22ff279` | 81.23 | 55.81 | 4467 | 1.4555x | -0.000% | 10/10 |
| v21 | `a67816b` | 80.84 | 55.48 | 4467 | 1.4571x | -0.001% | 10/10 |
| v22 | `7a8b38f` | 80.70 | 54.64 | 4467 | 1.4769x | +0.001% | 10/10 |
| v23 | `d8ac022` | 80.68 | 52.76 | 6747 | 1.5292x | -0.003% | 10/10 |
| v24 | `053c9c4` | 78.99 | 50.95 | 6651 | 1.5503x | -0.001% | 10/10 |
| v25 | `fd296f4` | 80.87 | 48.97 | 6651 | 1.6514x | +0.000% | 10/10 |
| v26 | `79643de` | 80.84 | 46.38 | 6651 | 1.7430x | -0.000% | 10/10 |
| v27 | `7efba21` | 80.79 | 45.83 | 7153 | 1.7628x | +0.002% | 10/10 |
| v28 | `4aaaade` | 80.75 | 44.43 | 7153 | 1.8175x | -0.000% | 10/10 |
| v34 | `fe2272c` | 78.68 | 43.91 | 7151 | 1.7918x | -0.001% | 10/10 |
| v35 | `bba28df` | 80.81 | 46.46 | 4651 | 1.7393x | +0.004% | 10/10 |
| v36 | `dd8ccef` | 80.96 | 44.39 | 7151 | 1.8238x | +0.001% | 10/10 |

Every row is an original two-base/two-solution T4 report. The complete source
file set, source bytes, original result checksum and all ten checks were verified
locally from the raw archives. `03_validation/reports/codex_VERSION/` holds proof,
result, check and console files. Adapter archives are exported separately.

Close medians overlap ordinary run-to-run variation; this is the shortest
observed result, not a guarantee that every rerun wins. Compare each solution
with its own paired base. Memory efficiency also affects hackathon points, so
the shortest-time candidate does not by itself establish the highest final rank.

The fixed task stays Qwen2.5-1.5B, NF4/double-quant, LoRA r16, batch 4, sequence
512, 256 original Kazakh instructions and 64 steps on one T4. No changes were
made to organizer harness/task/tests. No model or compiler cache is moved outside
the timed process. Own-kernel time is measured by the unchanged organizer code.

The target **20–30 seconds for the complete unchanged command is not achieved**.
No first-place or maximum-possible claim is made. GPU/CPU profiling and isolated
matrix microbenchmarks are diagnostics; their times are excluded from this table.
The inactive v29 GEMM research failed its strict numeric tests and remains with
those tests in the research folder. It is never used by the production solution.
v30/v31 were unrun; v32/v33 stopped before training on those research tests.
The separate 11-shape cuBLASLt screen achieved 0.9072x against the default
vendor BLAS (about 10.2% more kernel time), so no production backend flag changed.

Vladimir's shared forward/launcher work and Merei's ordered-gradient sums and
single-read batch metadata are explicitly credited in TEAM_INTEGRATION.md.
Failed initial environment runs and earlier candidates
remain in history. Final team selection and a public saved executed notebook are
still separate hackathon submission steps.

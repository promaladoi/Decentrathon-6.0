# v16: verified two-pair T4 result

Measured code: `4a5cfd20fd79e597e15919f3f19f959a6b4d689a`.
Downloaded exact archive: `codex_v16.zip`. Source files, the result SHA-256 and
the organizer checks' solution SHA-256 were verified against that commit locally.

| Metric | Base | v16 |
|---|---:|---:|
| Run 1, seconds | 82.06 | 60.26 |
| Run 2, seconds | 88.08 | 60.56 |
| Median, seconds | 85.07 | 60.41 |
| Peak memory, MiB | 14345 | 5601 |
| Train-loop median, seconds | 62.16 | 36.30 |
| Public eval loss | 2.04966 | 2.04968 |

Speedup **1.4082x** (40.8% higher throughput), **29.0% less total runtime**.
Memory ratio **2.5611x**, **61.0% less peak memory**. Eval loss **+0.001%**,
within the rule's 0.5% bound; do not describe this as an identical/lower loss.
**All ten rule checks passed**, 115 kernel tests and 79 Soup tests passed.
Own-kernel minimum share: **12.9%**. The fixed task, model/data, optimizer,
dropout, Soup switches and organizer harness/tests remain unchanged.

The corrected compact dropout handles noncontiguous row indices and preserves
exact output/gradient/RNG tests. The startup thread compiles the actual kernels
used by the parallel head and compact dropout while the model loads.
Vladimir's unpadding/normalization/rotary/NF4 path remains credited in
TEAM_INTEGRATION.md. The original organizer SwiGLU is excluded from own kernels.

Raw reports and logs: `03_validation/reports/codex_v16/`.
Reproduce: `templates/codex_v16.ipynb` (or `codex_candidate.ipynb`).
The small speedup-ratio difference from v13 is not conclusive evidence of an
incremental speed gain; separate-session variation exists. v16 is now the latest
fully collected and verified candidate. No 70–90% time reduction or victory is claimed.

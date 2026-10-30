# v46: original 10/10, rejected by the stricter user quality rule

Source `30075981acb89e26e8b0a2fa4659d8b4c100daed`; original archive downloaded
and verified against all 49 source files and original report hashes.

| Metric | Paired base | v46 |
|---|---:|---:|
| Full command, seconds | 79.69 | 45.75 |
| Peak GPU, MiB | 14345 | 4135 |
| Eval loss | 2.04963 | 2.04967 |
| Repeats | 2 | 2 |

All original ten checks passed; own kernels >=14.7%. Solution tests: 390 passed,
1 skipped (native metadata parity passed separately). Soup: 79 passed.
Native dropout/LoRA equivalence tests: 8 passed, including exact values, gradients
and RNG progression. These focused tests do not supersede whole-model evaluation.

The original task permits this +0.002% eval change, but the user's rule is stricter:
reported eval loss must not increase. Therefore v46 is **not selected**. It also
does not improve the recorded v45 time or peak. The verified v45 remains selected.
30 seconds is not reached. Candidate v47 inherited this cast and was superseded
before measurement; v48 restores exact v45 arithmetic and changes only lifetimes.

Original evidence: [reports](03_validation/reports/codex_v46),
[collection proof](03_validation/reports/codex_v46/collection_proof.json),
[user constraints](03_validation/reports/codex_v46/user_constraints.json).

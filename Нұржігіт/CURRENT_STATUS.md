# Current selection and latest measurements

| Version | Paired base, s | Solution, s | Peak MiB | Eval loss, base → solution | Original checks | User memory/quality |
|---|---:|---:|---:|---|---|---|
| v45 | 79.56 | 45.03 | 4135 | 2.04963 → 2.04960 | 10/10 | pass |
| v46 | 79.69 | 45.75 | 4135 | 2.04963 → 2.04967 | 10/10 | FAIL: loss increased |
| v48 | 81.01 | 46.59 | 4115 | 2.04968 → 2.04964 | 10/10 | pass |
| v49 | 80.08 | 45.69 | 4115 | 2.04967 → 2.04966 | 10/10 | pass |

Public main: selected v45, 45.03 seconds / 4135 MiB / original 10/10.
v48 and v49 are measured memory alternatives (4115 MiB), not replacements.
v46 failed the stricter nonincreasing-loss rule; v47 was unmeasured.
The current whole-command target is <=27 seconds and remains unmet.
Historical reports retain their original <=30-second target.

v50 native batched gate/up research was rejected before integration: 12/14
shape/seed cases changed FP16 values; the two exact cases were slower.
[Unmodified T4 evidence](03_validation/reports/v50_research/README.md).

v51 moves stable causal-target partitions into the existing metadata kernel.
Its first 70 target/metadata GPU tests passed; complete validation is running.
[Candidate and pending status](experiments/v51). No v51 speed or admission is claimed yet.

[Latest decisions and evidence](CODEX_OPTIMIZATION_RESULTS.md).
[Selected result](CODEX_V45_RESULT.md). [Selected source proof](PUBLICATION_PROOF.json).
Pinned reproducible notebook: `templates/codex_v45.ipynb`.
Development HEAD may contain experimental code; use the exact measured commit.

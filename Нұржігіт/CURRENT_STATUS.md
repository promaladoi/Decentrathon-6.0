# Current selection and latest measurements

| Version | Paired base, s | Solution, s | Peak MiB | Eval loss, base → solution | Original checks | User memory/quality |
|---|---:|---:|---:|---|---|---|
| v45 | 79.56 | 45.03 | 4135 | 2.04963 → 2.04960 | 10/10 | pass |
| v46 | 79.69 | 45.75 | 4135 | 2.04963 → 2.04967 | 10/10 | FAIL: loss increased |
| v48 | 81.01 | 46.59 | 4115 | 2.04968 → 2.04964 | 10/10 | pass |
| v49 | 80.08 | 45.69 | 4115 | 2.04967 → 2.04966 | 10/10 | pass |
| v51 | 84.20 | 49.03 | 4135 | 2.04964 → 2.04963 | 10/10 | pass |
| v53r1 | 83.14 | 47.51 | 4109 | 2.04961 → 2.04965 | 10/10 | FAIL: loss increased |

Public main: selected v45, 45.03 seconds / 4135 MiB / original 10/10.
v48 and v49 are measured memory alternatives (4115 MiB), not replacements.
v46 failed the stricter nonincreasing-loss rule; v47 was unmeasured.
The current whole-command target is <=27 seconds and remains unmet.
Historical reports retain their original <=30-second target.

v50 native batched gate/up research was rejected before integration: 12/14
shape/seed cases changed FP16 values; the two exact cases were slower.
[Unmodified T4 evidence](03_validation/reports/v50_research/README.md).

v51 moves stable causal-target partitions into the existing metadata kernel.
Original 10/10, 412 tests and the user memory/quality limits passed.
[Measured v51 result](CODEX_V51_RESULT.md). It is not selected; the 27-second target remains unmet.

[Same-session v45 control](03_validation/reports/v45_same_session_control/README.md):
48.55 seconds / 4135 MiB, solution-only diagnostic. No v51 speed advantage established.
v52/v52r1 explicit-MMA research matched sampled native values but was slower;
[both screens](03_validation/reports/v52r1_research/README.md) were kept inactive.

[Latest decisions and evidence](CODEX_OPTIMIZATION_RESULTS.md).
[Selected result](CODEX_V45_RESULT.md). [Selected source proof](PUBLICATION_PROOF.json).
Pinned reproducible notebook: `templates/codex_v45.ipynb`.
Development HEAD may contain experimental code; use the exact measured commit.

[v53/v53r1 verified outcome](CODEX_V53_RESULT.md): 47.51 s / 4109 MiB, original 10/10; user nonincreasing-loss requirement failed. Not selected.

[v54 native NF4 screen](03_validation/reports/v54_research): large MLP projections were slower; limited narrow-projection opportunity remains unintegrated. No new full-task score.

[v55 activation recomputation](experiments/v55): ten exact targeted GPU tests passed; full original validation running. Smaller saved activation storage is tested; no full-command speed or peak-memory result is claimed yet.

[v56 narrow NF4 dispatch](experiments/v56): separate v45-based speed candidate; pending serial qualification after v55. No result or selection claimed.

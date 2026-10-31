# Current selection and latest measurements

| Version | Paired base, s | Solution, s | Peak MiB | Eval loss, base → solution | Original checks | User memory/quality |
|---|---:|---:|---:|---|---|---|
| v45 | 79.56 | 45.03 | 4135 | 2.04963 → 2.04960 | 10/10 | pass |
| v46 | 79.69 | 45.75 | 4135 | 2.04963 → 2.04967 | 10/10 | FAIL: loss increased |
| v48 | 81.01 | 46.59 | 4115 | 2.04968 → 2.04964 | 10/10 | pass |
| v49 | 80.08 | 45.69 | 4115 | 2.04967 → 2.04966 | 10/10 | pass |
| v51 | 84.20 | 49.03 | 4135 | 2.04964 → 2.04963 | 10/10 | pass |
| v53r1 | 83.14 | 47.51 | 4109 | 2.04961 → 2.04965 | 10/10 | FAIL: loss increased |
| v55 | 83.80 | 52.23 | 3555 | 2.04964 → 2.04967 | 10/10 | FAIL: user limits |

Public main: selected v45, 45.03 seconds / 4135 MiB / original 10/10.
v48 and v49 are measured memory alternatives (4115 MiB), not replacements.
v46 failed the stricter nonincreasing-loss rule; v47 was unmeasured.
The current whole-command <=27-second target remains unmet.

[Latest decisions and evidence](CODEX_OPTIMIZATION_RESULTS.md).
[Selected result](CODEX_V45_RESULT.md). [Selected source proof](https://github.com/promaladoi/Decentrathon-6.0/blob/main/%D0%9D%D2%B1%D1%80%D0%B6%D1%96%D0%B3%D1%96%D1%82/PUBLICATION_PROOF.json).
Pinned reproducible notebook: `templates/codex_v45.ipynb`.
Development HEAD may contain experimental code; use the exact measured commit.

[Measured v51 result](CODEX_V51_RESULT.md). No confirmed improvement; v45 stays selected.

Same-session v45 solution-only control: 48.55 s / 4135 MiB; v51: 49.03 s.
v52/v52r1 explicit-MMA prototypes matched sampled native values but were slower
and were not integrated. See `03_validation/reports/v52r1_research/README.md`.

[v53/v53r1 verified outcome](CODEX_V53_RESULT.md): not selected; user quality requirement failed.

[v54 native NF4 screen](03_validation/reports/v54_research): large MLP projections were slower; limited narrow-projection opportunity remains unintegrated. No new full-task score.

[v55 measured outcome](CODEX_V55_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

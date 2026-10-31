# Current selection and latest measurements

| Version | Paired base, s | Solution, s | Peak MiB | Eval loss, base → solution | Original checks | User memory/quality |
|---|---:|---:|---:|---|---|---|
| v45 | 79.56 | 45.03 | 4135 | 2.04963 → 2.04960 | 10/10 | pass |
| v46 | 79.69 | 45.75 | 4135 | 2.04963 → 2.04967 | 10/10 | FAIL: loss increased |
| v48 | 81.01 | 46.59 | 4115 | 2.04968 → 2.04964 | 10/10 | pass |
| v49 | 80.08 | 45.69 | 4115 | 2.04967 → 2.04966 | 10/10 | pass |
| v51 | 84.20 | 49.03 | 4135 | 2.04964 → 2.04963 | 10/10 | pass |

Public main: selected v45, 45.03 seconds / 4135 MiB / original 10/10.
v48 and v49 are measured memory alternatives (4115 MiB), not replacements.
v46 failed the stricter nonincreasing-loss rule; v47 was unmeasured.
The current whole-command <=27-second target remains unmet.

[Latest decisions and evidence](CODEX_OPTIMIZATION_RESULTS.md).
[Selected result](CODEX_V45_RESULT.md). [Selected source proof](https://github.com/promaladoi/Decentrathon-6.0/blob/main/%D0%9D%D2%B1%D1%80%D0%B6%D1%96%D0%B3%D1%96%D1%82/PUBLICATION_PROOF.json).
Pinned reproducible notebook: `templates/codex_v45.ipynb`.
Development HEAD may contain experimental code; use the exact measured commit.

[Measured v51 result](CODEX_V51_RESULT.md). No confirmed improvement; v45 stays selected.

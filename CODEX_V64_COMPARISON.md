# v64r2: verified lower-memory alternative

| Variant in the same Kaggle session | Median whole command | NVML peak |
|---|---:|---:|
| Exact v45 control | 47.27 s | 4135 MiB |
| v64r2 | 47.61 s | 3915 MiB |

v64r2 saves **220 MiB (5.32%)**, with an observed **0.34 s (0.72%)** increase in median whole-command time. This does not establish a speed improvement. Two repetitions per variant leave timing uncertainty.

v64r2 passed the original **10/10** checks with two base and two solution runs. Its paired base was 86.33 s / 14345 MiB; reported evaluation loss was **2.04966 -> 2.04963**, satisfying the additional nonincrease requirement. The separate exact-v45 solution-only control is not a new ten-check admission.

Individual v45 control times: **45.87 and 48.68 s**. Individual v64r2 times: **46.17 and 49.04 s**. All repeats kept their variant's stated peak.

**Decision:** retain exact measured v45 as the selected implementation. Keep the fully verified v64r2 under `experiments/v64r2` as the lower-memory alternative. The **<=25-second whole-command target is unmet**. The historical 45.03-second v45 paired report belongs to an earlier session.

[Original paired v64r2 proof](03_validation/reports/codex_v64r2/collection_proof.json). [Same-session v45 control](03_validation/reports/v45_control_after_v64r2/README.md). [Selection/source comparison](04_submission/v64_revisions_selection.json).

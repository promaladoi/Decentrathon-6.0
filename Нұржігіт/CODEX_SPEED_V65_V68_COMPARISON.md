# Whole-command speed comparison: v65-v68

| Variant | Whole command | Peak MiB | Reported loss, base -> solution | Original checks | User memory/quality |
|---|---:|---:|---|---:|---|
| v45 current solution-only control | 48.23 s | 4135 | separate control | not a new admission | timing control |
| v65 | 43.77 s | 6659 | 2.04965 -> 2.04966 | 10/10 | FAIL |
| v66 | 46.20 s | 7039 | 2.04965 -> 2.04964 | 10/10 | pass |
| v67 | 48.50 s | 4135 | 2.04966 -> 2.04968 | 10/10 | FAIL |
| v68 | 46.20 s | 6879 | 2.04967 -> 2.04963 | 10/10 | pass |

Selected **v45**, exact source `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`. No new candidate justifies replacing the selected v45 under all requested constraints. The **<=25-second whole-command target is not met**.

The control is solution-only, not a new ten-check admission. Original per-run metrics, logs, checks and source hashes are retained. Each candidate must independently satisfy the original ten checks, <=7151 MiB for every run and nonincreasing reported evaluation loss. Small timing/loss variations are shown rather than hidden.

[Selection/source verification](04_submission/speed_v65_v68_selection.json). [Current v45 control](03_validation/reports/v45_control_after_v68/README.md).

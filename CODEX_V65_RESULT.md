# v65 verified original outcome

Exact source: `0d12b9ee8aada7269c11bf1c5cdea776f305477a`; fifty Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 81.69 s | 14345 MiB | 2.04965 |
| v65 | 43.77 s | 6659 MiB | 2.04966 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 24.88 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: FAIL.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v65/collection_proof.json). [User limits](03_validation/reports/codex_v65/user_constraints.json). [Candidate description](CODEX_V65_EXPERIMENT.md).

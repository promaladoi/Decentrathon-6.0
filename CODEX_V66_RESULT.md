# v66 verified original outcome

Exact source: `507d468e40bd6dc52cd92cdba6ce00d6f1771df4`; fifty Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 82.67 s | 14345 MiB | 2.04965 |
| v66 | 46.20 s | 7039 MiB | 2.04964 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 27.33 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: pass.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v66/collection_proof.json). [User limits](03_validation/reports/codex_v66/user_constraints.json). [Candidate description](CODEX_V66_EXPERIMENT.md).

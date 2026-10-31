# v72 verified original outcome

Exact source: `1683602834721b809a5983bb096714705c66f719`; 56 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 84.00 s | 14345 MiB | 2.04966 |
| v72 | 48.61 s | 3959 MiB | 2.04964 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 29.34 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: pass.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v72/collection_proof.json). [User limits](03_validation/reports/codex_v72/user_constraints.json). [Candidate description](CODEX_V72_EXPERIMENT.md).

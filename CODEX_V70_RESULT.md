# v70 verified original outcome

Exact source: `878afbc5d0d7a1a6ce90d62b7737f409bc65cb8c`; 54 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 82.53 s | 14345 MiB | 2.04965 |
| v70 | 48.30 s | 3959 MiB | 2.04968 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 29.00 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: FAIL.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. It adds a qualified private native-Philox coordinate gather to v69, retaining the complete padded RNG counter advance and native fallback for unqualified consumers. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v70/collection_proof.json). [User limits](03_validation/reports/codex_v70/user_constraints.json). [Candidate description](CODEX_V70_EXPERIMENT.md).

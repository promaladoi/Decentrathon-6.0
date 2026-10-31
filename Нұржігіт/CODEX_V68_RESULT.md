# v68 verified original outcome

Exact source: `46fea8a4d636e8cb24c1a1522d788443cf363a79`; 52 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 84.17 s | 14345 MiB | 2.04967 |
| v68 | 46.20 s | 6879 MiB | 2.04963 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 27.34 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: pass.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. It combines the guarded CLI startup with full eligible residency, retaining the 512-row head chunk. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v68/collection_proof.json). [User limits](03_validation/reports/codex_v68/user_constraints.json). [Candidate description](CODEX_V68_EXPERIMENT.md).

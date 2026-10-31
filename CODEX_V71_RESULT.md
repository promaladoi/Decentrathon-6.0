# v71 verified original outcome

Exact source: `d8194784a48b833e7b79edf4bbc8bd2daa1d8d2f`; 54 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 85.33 s | 14345 MiB | 2.04967 |
| v71 | 49.06 s | 3959 MiB | 2.04962 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 29.61 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: pass.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. It qualifies the private RNG consumer independently of AMP gradient-scale availability; both available and unavailable scales preserve exact values, gradients and RNG in unit qualification. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v71/collection_proof.json). [User limits](03_validation/reports/codex_v71/user_constraints.json). [Candidate description](CODEX_V71_EXPERIMENT.md).

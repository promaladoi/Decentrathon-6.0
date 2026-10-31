# v69 verified original outcome

Exact source: `6d91d6430f88b5786ba905fc931ff7a9de164c2d`; 52 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 82.05 s | 14345 MiB | 2.04964 |
| v69 | 46.73 s | 3959 MiB | 2.04962 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 27.59 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: pass.

This is a time-priority, bounded-residency candidate. Initial decoding and allocation remain inside each fresh measured subprocess. It combines v64r2 LoRA-input rematerialization with a 64 MiB cache restricted to small frozen matrices, retaining the original 512-row head chunk and CLI. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v69/collection_proof.json). [User limits](03_validation/reports/codex_v69/user_constraints.json). [Candidate description](CODEX_V69_EXPERIMENT.md).

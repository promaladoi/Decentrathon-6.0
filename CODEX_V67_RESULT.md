# v67 verified original outcome

Exact source: `4f205475601503bd2467f5785c5483bd58bbb3c5`; 50 Python files verified byte for byte against Git.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 84.38 s | 14345 MiB | 2.04966 |
| v67 | 48.50 s | 4135 MiB | 2.04968 |

Original checks: **10/10**, with two base and two solution runs. Training loop: 29.61 s; this is only part of the measured whole command.

The <=25-second whole-command target is not met.

Every solution <=7151 MiB: pass. Nonincreasing reported eval loss: FAIL.

This is the exact-source-guarded CLI startup candidate on unchanged v45 training math. No resident NF4 weight cache, rematerialization or larger head chunk is included. No data, optimizer, training-step or measurement-boundary change is included.

Selection remains the previously verified implementation until a completed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v67/collection_proof.json). [User limits](03_validation/reports/codex_v67/user_constraints.json). [Candidate description](CODEX_V67_EXPERIMENT.md).

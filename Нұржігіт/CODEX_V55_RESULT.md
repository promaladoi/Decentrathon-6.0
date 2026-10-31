# v55 verified original outcome

Exact source: `1569729553dd219249cf95c17ca37c865917fac7`; 49 Python files verified against Git byte for byte.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 83.80 s | 14345 MiB | 2.04964 |
| v55 | 52.23 s | 3555 MiB | 2.04967 |

Original checks: **10/10**. Two base and two solution runs; full raw per-run metrics and logs are retained.

Training-loop time: 32.51 s, distinct from the whole command. The <=27-second whole-command target is not met.

The stricter user memory/quality rule does not pass; inspect user_constraints.json.

This is a memory/time tradeoff. Recomputing the same frozen up activation reduces saved storage, but adds arithmetic. It is retained as an experimental memory alternative; the selected faster v45 remains unchanged.

[Source/result collection proof](03_validation/reports/codex_v55/collection_proof.json).
[User limits](03_validation/reports/codex_v55/user_constraints.json).

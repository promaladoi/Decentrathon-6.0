# v63 verified original outcome

Exact source: `48cef7c3fe1993ef699b96b7f2336ca795fb5a4d`; 49 Python files verified against Git byte for byte.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 83.68 s | 14345 MiB | 2.04962 |
| v63 | 48.69 s | 4115 MiB | 2.04963 |

Original checks: **10/10**. Two base and two solution runs; full raw per-run metrics and logs are retained.

Training-loop time: 28.63 s, distinct from the whole command. The <=27-second whole-command target is not met.

The stricter user memory/quality rule does not pass; inspect user_constraints.json.

This candidate keeps v45 arithmetic and tunes the same own NF4 kernel from the v62 screen. Dead-reference lifetime changes come from the earlier verified v48. No CUDA graph or experimental shared MMA is installed in training. Selection stays exact measured v45. v63 does not establish a whole-command improvement and fails the stricter nonincreasing reported-loss requirement.

[Source/result collection proof](03_validation/reports/codex_v63/collection_proof.json).
[User limits](03_validation/reports/codex_v63/user_constraints.json).

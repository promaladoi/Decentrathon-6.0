# v56 verified original outcome

Exact source: `1dc91c6a7cf7ed73baf42c1519ea393757a03bef`; 49 Python files verified against Git byte for byte.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 85.69 s | 14345 MiB | 2.04960 |
| v56 | 49.69 s | 4135 MiB | 2.04968 |

Original checks: **10/10**. Two base and two solution runs; full raw per-run metrics and logs are retained.

Training-loop time: 30.02 s, distinct from the whole command. The <=27-second whole-command target is not met.

The <=7151 MiB memory bound passes in both solution runs. Reported eval loss increased by +0.004%, so the stricter nonincreasing-loss user rule fails even though the original quality tolerance passes.

This geometry policy calls the already-installed vendor bitsandbytes backend. It is not credited as team-authored kernel code. Selection stays v45; no confirmed overall advantage justifies promotion.

[Source/result collection proof](03_validation/reports/codex_v56/collection_proof.json).
[User limits](03_validation/reports/codex_v56/user_constraints.json).

Validation: 9 targeted tests; 391 solution tests passed / 1 skipped; 79 Soup regression tests passed; native metadata parity passed separately.

[Original archive and checksum index](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v56-20261031).

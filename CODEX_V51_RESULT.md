# v51 verified original result

Source: `10b8f2a71a85de2863399d351f043c92073ca305` (49 Python files verified byte for byte).

| Side | Median whole command | Peak GPU memory | Eval loss |
|---|---:|---:|---:|
| Base | 84.20 s | 14345 MiB | 2.04964 |
| v51 | 49.03 s | 4135 MiB | 2.04963 |

Two original base and two solution runs; original **10/10 PASS**.
Solution times were 48.08 and 49.99 seconds. Original own-kernel coverage is
14.0% or more. Full suite: 412 passed, 1 skipped; native metadata parity passed
separately; Soup regression suite: 79 passed. No original check was waived.

The <=7151 MiB bound and nonincreasing reported eval-loss requirement pass.
The <=27-second whole-command target is **not met**. The training-loop median
is 28.69 s and must not be presented as the whole-command time.

Public selection stays v45 (historical 45.03 s / 4135 MiB / 10/10).
Same-session solution-only v45 control measured 48.55 s / 4135 MiB, versus
v51 49.03 s. The 0.48 s difference does not establish any speed advantage
for v51; timing noise remains. The control is not a new ten-check admission.
[Control evidence](03_validation/reports/v45_same_session_control/README.md).

[Exact collection proof](03_validation/reports/codex_v51/collection_proof.json).
[Original checks](03_validation/reports/codex_v51/checks.json).
[User constraints](03_validation/reports/codex_v51/user_constraints.json).

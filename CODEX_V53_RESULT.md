# v53 / v53r1 verified result

v53 (373c904735bc5061c7ea06dd3cad553ee695be4c) stopped at a team-owned
test's obsolete small-batch routing expectation: 1 failed, 387 passed,
1 skipped. Its value, gradient and native RNG checks passed. No training or
original ten-check result was produced for that source.

v53r1 (0e32a5a2d0fbf79d7eedf5c2c5730fcc496fc4ab) updates that routing
expectation and adds sizes. Existing zero-tolerance numeric and RNG assertions
remain intact. 49 Python source files were verified against Git byte for byte.

| Side | Median whole command | NVML peak | Eval loss |
|---|---:|---:|---:|
| Base | 83.14 s | 14345 MiB | 2.04961 |
| v53r1 | 47.51 s | 4109 MiB | 2.04965 |

Two base and two solution runs. Solutions: 46.84 and 48.18 s, both 4109 MiB.
Original **10/10 PASS**, own kernels >=15.0%, 391 tests passed/1 skipped,
and 79 Soup regression tests passed. The loop time is 27.71 s; it is not the
whole-command time and does not meet the 27-second target.

The stricter user requirement of nonincreasing reported eval loss failed
(+0.002%). Memory dropped 26 MiB from the selected v45's 4135 MiB, but there
is no confirmed speed benefit. **Not selected. v45 remains selected.**

[Original proof](03_validation/reports/codex_v53r1/collection_proof.json).
[Original checks](03_validation/reports/codex_v53r1/checks.json).
[User limits](03_validation/reports/codex_v53r1/user_constraints.json).

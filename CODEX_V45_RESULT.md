# v45: original T4 result verified locally

Measured code: `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`. Collection completed 2026-10-31 (Asia/Qyzylorda).
The downloaded original archive matches the participant's SHA256, all 48
Python source files match Git byte for byte, and the original checks bind the
result bytes and complete solution digest. All ten original checks passed.

| Metric | Base | v45 |
|---|---:|---:|
| Complete command, seconds | 79.56 | 45.03 |
| Peak GPU, MiB | 14345 | 4135 |
| Training loop, seconds | 58.38 | 26.54 |
| Public evaluation loss | 2.04963 | 2.04960 |
| Paired runs | 2 | 2 |

Speedup 1.7668x; memory ratio 3.4692x. Relative to its paired base, complete
time falls 43.4% and memory 71.2%. Every solution run peaks at 4135 MiB,
below the user's 7151 MiB limit; reported eval loss does not increase.
Loss-curve, Soup outputs, clean-device and no-prohibited-library checks pass.
Own recurring kernels account for at least 14.6% of training time.

The 30-second whole-command target is NOT met: 15.03 seconds remain.
The all_requested_constraints_met=false field refers to that remaining target;
the original ten checks and the user's memory/quality limits passed.

## Tests in the original archive

- Dropout: 13 passed.
- Frozen-storage integration: 4 passed.
- Real native metadata parity: 1 passed (13 unrelated tests deselected).
- Head lifetime/scaled-gradient gate: 84 passed.
- Complete solution suite: 382 passed, 1 skipped.
- Original Soup regression suite: 79 passed, then 79 passed in rule check 10.

These are overlapping suites, not an additive unique-test count. The full
suite's opt-in native parity skip is covered by its separate explicit run.
PyTorch deprecation warnings did not cause failures. The complete-command
measurement includes its own startup and compilation, as required.

## Comparison and attribution

Compared with the earlier v34 report (43.91 seconds / 7151 MiB), v45 uses
3016 MiB less (42.2%), with an observed 1.12-second longer command (2.6%).
These were separate sessions, so this is not a same-session causal A/B.
Do not attribute the full memory gain to v45's tiny lifetime change: v44
already integrated Merei's corrected path and Nurzhigit's frozen storage.
No v44 full-task report has been collected here, so the incremental v44-to-v45
benefit is not isolated. Merei/Vladimir contributions retain their authorship.

v45 is the selected combined memory/speed candidate in Nurzhigit's folder.
v34 is retained as a historical fast alternative. This does not establish a
contest ranking, hidden-data acceptance, completion of external architecture
review, or final platform submission.

Original archive SHA256: `a7c06beaed9f3d9a45f1996100f4a09bdc97f2f28dc9ee8a19458adbcf1609cf`.
Solution SHA256: `8568c6f9abfb6458bfa4248c044cc2cf0169c104a487f8aeb4af4ae67702a787`.
Evidence: `03_validation/reports/codex_v45/collection_proof.json`, original
`result.json`, `checks.json`, `user_constraints.json` and per-run logs/metrics.
The ephemeral signed download URL is deliberately omitted from public logs.

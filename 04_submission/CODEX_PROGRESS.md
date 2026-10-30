# Codex work visible to the team

The team's request is to commit and push each completed code change and each
completed validation result, so Anais, Vladimir and Merey can follow the work.
Do not commit credentials, Kaggle secrets, model downloads or unfinished benchmark
claims. A commit describes a real change; empty activity commits are unnecessary.

## Where to follow

- Collaboration mirror: https://github.com/promaladoi/Decentrathon-6.0/tree/codex/kernel-v1
- Commit history: https://github.com/promaladoi/Decentrathon-6.0/commits/codex/kernel-v1
- Hackathon platform repository: https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai/tree/codex/kernel-v1

Both repositories receive the same `codex/kernel-v1` branch. The platform
repository remains the development and eventual submission source. Neither
repository's `main` branch is changed by this mirror; teammates' folders are
preserved. The winning version is still to be selected by the team.

## Completed milestones

| Code commit | Change | Measured outcome |
|---|---|---|
| `d4d4231` | Original tiled vocabulary and frozen RMSNorm kernels | 9/10 checks; own kernels only 3.3% of training time |
| `6c7370c` | Original NF4 decoder in backward | 10/10 checks; own-kernel share 5.1% |
| `8f82109` | Use the decoder in the existing forward fallback | 10/10 checks; own-kernel share 7.1% |
| `3f86aa3` | Reproduction notebook, raw reports and comparison instructions | Code and raw-report hashes verified |

Latest validated T4 result, two paired runs: base **88.08 s**, candidate
**85.30 s**; peak memory **14345 / 12385 MiB**; eval loss **2.04963 / 2.04963**.
The measured speedup is **1.0326x**, below the competition's 5% speed-scoring
threshold. This is a validated candidate, not a claim of maximum possible speed.
The retained organizer SwiGLU example accounts for much of the memory saving.

Read `CODEX_VARIANT.md` for implementation details and
`03_validation/VERSION_COMPARISON.md` for the version comparison.

## Next work

The unchanged starter verification notebook is
`templates/original_starter_kaggle.ipynb`. Import it into Kaggle, select GPU T4 x2
and Internet On, then Run All. It pins the organizer commit and uses no GitHub
token. Checks 3 and 4 are expected to fail for the organizer example. A direct
verification of organizer commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`
finished on Kaggle T4: base **79.03 s / 14345 MiB / loss 2.04961**, starter
solution **86.24 s / 12385 MiB / loss 2.04968**. The four starter kernel tests
and 79 Soup regression tests passed. Checks 3 and 4 failed as expected; the other
eight checks passed. This is one paired run, not a directly comparable absolute
time against a different experiment. Use the paired speedup ratio.

Experimental code `12b9f58` avoids a redundant float32 decoded-weight buffer
when the existing linear operation already uses CUDA fp16 autocast. Additional
tests compare forward values and input gradients with the original backend,
with and without autocast. T4 validation finished: **37 kernel tests, 79 Soup
tests and 10/10 checks passed**. Paired medians: **86.17 s base / 83.18 s v4**;
speedup **1.0359x**, memory ratio **1.1583x**, own-kernel share **7.1%**.
This does not demonstrate a meaningful speed gain over v3. See
`CODEX_V4_EXPERIMENT.md`; all original raw reports are committed under
`03_validation/reports/`. The final team version has not been selected.

Investigate remaining training-time bottlenecks. Record each finished experiment, including failures,
with its exact code commit and original benchmark/check reports. Compare other
team members' versions before choosing the final submission.

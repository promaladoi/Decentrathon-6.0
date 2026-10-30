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

Verify the unchanged public starter directly, then investigate remaining
training-time bottlenecks. Record each finished experiment, including failures,
with its exact code commit and original benchmark/check reports. Compare other
team members' versions before choosing the final submission.

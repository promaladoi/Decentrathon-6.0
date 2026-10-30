# Latest: v16 verified; v17/v18/v20 queued experiments

- **Fully verified v16:** code `4a5cfd2`, 2 paired T4 runs, 85.07 / 60.41 s,
  speedup 1.4082x, memory 14345 / 5601 MiB (61.0% less), eval loss +0.001%,
  115 kernel tests + 79 Soup tests and all 10 checks. Own share 12.9%.
  Exact raw-report/source hashes and adapter archive were collected and verified.
- **Branch-tip solution is an experiment, not the validated v16 source.**
  Reproduce/export v16 using its pinned notebook or exact commit, not branch tip.
- **Pending v17:** run snapshot `7c5f6d8`, paired-byte NF4 reconstruction.
  104 CPU address/layout cases pass; 23 new GPU cases and microbenchmark await T4.
- **Pending v18:** code `d01b2d8`, lossless frozen tied embedding/head storage.
  15 CPU contract tests pass; 5 CUDA cases skipped locally. Full GPU suite 158.
- **v19 intermediate:** code `1e2fe4b`, native dropout mask kept directly;
  no independent measurement. Its six new exact mask/RNG cases remain in v20.
- **Pending v20:** code `22ff279`, only real-token dropout bits saved, 8 per byte.
  124 bounds-checked CPU kernel-body layout cases pass. Dropout GPU preflight 24,
  full solution suite 174; CUDA compilation, RNG, quality and speed unverified.
- The finite coordinator queues v17, v18 and v20, serially, max 3 jobs/50 minutes.
  Every worker uses a separate exact checkout, unchanged organizer files, strict
  correctness gates, 2 paired measurements, quality evaluation and all ten checks.
  It fetches queued revisions only between workers, never during measurement.
- **Coordinator not confirmed started:** the known Kaggle status artifact returns
  HTTP 404. Browser control is disconnected and no local Kaggle API credential
  is configured. `templates/run_experiments_cell.py` is the one initial launch
  cell for the existing v16 Kaggle environment; no other manual edits are needed.
- 70–90% total-time reduction and a first-place result have not been achieved.
- Earlier v13 console, v12 archived fallback and failed v14/v15 evidence remain.

## Earlier progress records

# Latest: v16 fully verified; v17 decoder and bounded coordinator prepared

- **Fully verified v16:** code `4a5cfd2`, 2 paired T4 runs, 85.07 / 60.41 s,
  speedup 1.4082x, memory 14345 / 5601 MiB (61.0% less), eval loss +0.001%,
  115 kernel tests + 79 Soup tests and all 10 checks. Own share 12.9%.
  Exact raw-report/source hashes and adapter archive were collected and verified.
- v13 remains an earlier accepted console result; v12 is an archived fallback.
  v14/v15 failed exact dropout tests and never trained; evidence is preserved.
- **Pending v17:** code `2c2baf4`, run snapshot `7c5f6d8`. Paired-byte NF4
  reconstruction with the same values and no persistent dense cache. 104 CPU
  address/layout cases pass, but GPU parity/quality/performance are unverified.
  It adds 23 GPU cases (138 total) and an independent decoder microbenchmark.
- The finite coordinator can run exact queued revisions serially, using separate
  checkouts and the unchanged task, with no polling during measurement. It needs
  one initial launch in Kaggle; no coordinator launch is claimed yet.
- 70–90% total-time reduction and a first-place result have not been achieved.


# Latest: v15 failure source-verified, row-stride bug repaired in v16

- Accepted console result: v13 `13eaa8a`, 1.398x, 5761 MiB, eval 2.04966,
  106 kernel tests + 79 Soup tests, all 10 checks. Raw v13 archive pending.
- Fully archived fallback: v12 `bd2b49a`, 1.1608x, 7385 MiB, all 10 checks;
  exact raw-report and source hashes verified locally.
- v14 `fd5e482`: 4 tests failed; no training. v15 `389f037`: 6 tests failed;
  no training. v15 archive downloaded; 31 source files match Git exactly.
- Actual issue: strided row-index views were read as dense indices. The earlier
  claim that RNG mapping differed was not established. v16 normalizes row storage
  once and saves it for backward. All strict numerical/RNG tests remain.
- v16 source: `4a5cfd20fd79e597e15919f3f19f959a6b4d689a`. It retains startup compilation, adds 2 layout
  regression cases (8 dropout / 115 total kernel tests), and improves failed
  outcome metadata. GPU tests, quality and performance are NOT yet verified.
- Use `templates/run_v16_cell.py` after the failed v15 cell. It preserves the
  existing data and prior run outputs, selects the exact commit and runs the
  guarded validator with a new `runs_codex_v16` output directory.
- Browser automation remains disconnected/blocked. No autonomous v16 run is claimed.


# Latest: v14 failure caught before training; repair and v15 prepared

- **Accepted console result: v13** (`13eaa8a`): 1.398x speedup, 5761 MiB,
  eval loss 2.04966, 106 kernel tests + 79 Soup tests, 10/10 checks.
  Final output supplied by participant; raw archive still awaits collection.
- **Fully archived fallback: v12** (`bd2b49a`): 1.1608x, 7385 MiB, 10/10,
  source and report hashes verified locally.
- **Rejected v14** (`fd5e482`): 4 exact dropout tests failed, 106 passed.
  RNG state matched but masks did not. Benchmark stopped before training.
  Raw supplied log is under `03_validation/reports/codex_v14_failed/`.
- **v14 repair** (`3016d55`): uses the original functional.dropout operation
  to produce a boolean mask; includes 6 exact-value/gradient/RNG-sequence tests.
  Pending GPU validation. No speed or quality claim.
- **v15 experiment** (`ef10dd1` computation): includes the mask repair and
  compiles the actual parallel-head/dropout kernels while the model loads.
  Pending 113 kernel tests, 79 Soup tests, two paired runs and all 10 checks.
  Syntax/whitespace validation passed locally; local Python has no CUDA/Triton.
- `04_submission/run_candidate.py` checks untouched task files and an idle T4,
  then runs dropout tests before the full suite and benchmark. A failed test
  stops training and still produces a source/log archive. No fake result JSON.
- Browser automation remains disconnected/blocked. The participant started v14
  manually and supplied its failing test output. No autonomous v15 run is claimed.
- Best-known team peer summary is Vladimir 1.468x / 5641 MiB (user supplied).
  No 70–90% reduction, victory or hidden-data quality guarantee is claimed.


# Latest: participant confirms v13 10/10

- **v13 completed:** code `13eaa8a025752db94a020102fa3ecd90951a2fec`.
  The participant supplied full console output with 106 kernel tests, 79 Soup tests,
  all 10 checks, speedup 1.398x, 5761 MiB and eval loss 2.04966. Own share 11.3%.
  Raw JSON, source archive and adapter still need to be downloaded/hash-verified.
- **v12 fully archived fallback:** code `bd2b49a`, speedup 1.1608x, 7385 MiB,
  64 kernel tests, 79 Soup tests, 10/10 checks, raw hashes/source verified locally.
- **v14 NOT confirmed launched:** code `fd5e4826645b85106859cae7a254297e52597fbf`.
  Browser control disconnected. The participant says no v14 cell is present.
  A fresh pinned `templates/codex_v14.ipynb` is prepared. It first tests exact
  dropout outputs, gradients and RNG state, then all 110 kernel tests, 79 Soup
  tests and two paired benchmark runs. No v14 performance or quality is claimed.
- The fresh notebook uses a new unique checkout, public mirror, and no GitHub
  token. Existing runs are preserved. It can also export existing v13 artifacts.
- A public saved final run remains necessary after the team selects a submission.
- No 70–90% improvement is claimed. Vladimir's user-reported 1.468x / 2.543x result
  is currently stronger than the reported v13 ratios; use a controlled comparison.


# Current update

The latest fully collected result is v12: 1.1608x speedup, 48.5% less peak memory,
64 kernel tests, 79 Soup tests and 10/10 checks. v13 combines Vladimir's shared
unpadded forward with Nurzhigit/Codex parallel vocabulary statistics; 106 kernel
tests and 79 Soup tests passed. Visible paired training medians: 88.52 / 63.31 s,
14345 / 5761 MiB. Its final evaluation and check report are still uncollected.
v14 preserves native dropout RNG but applies the mask only to real tokens.
The browser connection was lost during final evaluation of v13; no completed
v13/v14 result is claimed until exact artifacts have been collected and checked.

# Latest update

v10 passed all 10 checks: 1.1156x speedup, 48.5% less peak memory.
The v11/v12 code is committed before GPU verification, clearly experimental.
v12 has passed 64 kernel tests and is receiving a clean two-pair benchmark.
Work and reports are mirrored to the collaboration repository main branch under
`Нұржігіт/`. Raw reports are never edited to pass checks.

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

## Current validated result

v9 code `57182a2`: **1.1075x speedup**, **9.7% less runtime**, **36.1% less memory**.
Base / solution: **87.39 / 78.91 s**, **14345 / 9165 MiB**.
10/10 checks, 41 kernel tests, 79 Soup tests; own-kernel share 7.3%.
See [full report](../CODEX_V9_RESULT.md) and [pinned notebook](../templates/codex_candidate.ipynb).

v5 through v9 each have source commits and separate raw-report commits on both remotes.
v7's slower LoRA fusion was rejected; v9 retains the useful causal-loss memory improvement.
The interactive draft has a save conflict; downloaded artifacts are preserved independently.

## Completed milestones

| Code commit | Change | Measured outcome |
|---|---|---|
| `d4d4231` | Original tiled vocabulary and frozen RMSNorm kernels | 9/10 checks; own kernels only 3.3% of training time |
| `6c7370c` | Original NF4 decoder in backward | 10/10 checks; own-kernel share 5.1% |
| `8f82109` | Use the decoder in the existing forward fallback | 10/10 checks; own-kernel share 7.1% |
| `3f86aa3` | Reproduction notebook, raw reports and comparison instructions | Code and raw-report hashes verified |

Archived v3 T4 result, two paired runs: base **88.08 s**, candidate
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

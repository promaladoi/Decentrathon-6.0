# Verified optimization results: v45 through v49

Selected public `Нұржігіт/solution/` remains exact v45. The development branch
contains later experimental source; its moving HEAD is not the selected result.
Every measured row below has two base and two solution runs, original 10/10,
and a locally verified archive/source/checksum chain.

| Version | Paired base, s | Solution, s | Peak MiB | Eval loss, base → solution | Original checks | User memory/quality |
|---|---:|---:|---:|---|---|---|
| v45 | 79.56 | 45.03 | 4135 | 2.04963 → 2.04960 | 10/10 | pass |
| v46 | 79.69 | 45.75 | 4135 | 2.04963 → 2.04967 | 10/10 | FAIL: loss increased |
| v48 | 81.01 | 46.59 | 4115 | 2.04968 → 2.04964 | 10/10 | pass |
| v49 | 80.08 | 45.69 | 4115 | 2.04967 → 2.04966 | 10/10 | pass |

The <=30-second whole-command target is not met. The best observed memory among
these candidates is 4115 MiB (v48/v49), only 20 MiB below v45's 4135 MiB. Their
observed total times did not improve on v45. Separate runs are not a simultaneous
candidate-to-candidate A/B test, and differences this small may include noise.
The small memory tradeoff does not establish a stronger overall candidate.

- v46: original 10/10 passed, but reported eval loss increased. Rejected under
  the user's stricter quality requirement; no waiver or threshold change.
- v47: superseded before measurement because it inherited the v46 dropout cast.
- v48: restored exact v45 arithmetic and only released dead Python temporaries.
- v49: moved existing lossless frozen tied-storage packing earlier. Both raw
  training logs confirm the early path ran; the measured peak remained 4115 MiB.
- NF4 launch tuning: isolated timings were inconsistent; original launch kept.
- Smaller vocabulary chunks: some strict gradient checks failed; qualifying
  cases were slower. Production chunk size stayed unchanged.

All 15 protected organizer files remain unchanged. Profiling and isolated kernel
timings are diagnostics, not admitted contest results. No package/dataset/seed/
optimizer/step/allocator changes or cached work outside the timer are claimed.

For exact measured source, see `03_validation/reports/codex_VERSION/collection_proof.json`
and each `CODEX_VERSION_RESULT.md`. Historical alternatives (including v34)
remain archived. `PUBLICATION_PROOF.json` identifies selected source. Final
platform submission and a public saved executed Kaggle notebook are separate.

# Verified optimization results: v45 through v51

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
| v51 | 84.20 | 49.03 | 4135 | 2.04964 → 2.04963 | 10/10 | pass |
| v53r1 | 83.14 | 47.51 | 4109 | 2.04961 → 2.04965 | 10/10 | FAIL: loss increased |
| v55 | 83.80 | 52.23 | 3555 | 2.04964 → 2.04967 | 10/10 | FAIL: user limits |
| v56 | 85.69 | 49.69 | 4135 | 2.04960 → 2.04968 | 10/10 | FAIL: user limits |

The current <=27-second whole-command target is not met. Historical reports keep their original target. The best observed memory among
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

- v50: native BMM research rejected before integration (12/14 numerical cases failed).
- v51: exact target partitions passed original 10/10 and strict user memory/quality, but measured 49.03 s / 4135 MiB. same-session v45 control was 48.55 s / 4135 MiB; no speed advantage was established.

- v52/v52r1: explicit T4 MMA mapping and sampled FP16 parity passed, but both inactive prototypes were slower than native GEMM. Neither was integrated.

[v53/v53r1 verified outcome](CODEX_V53_RESULT.md): not selected; user quality requirement failed.

[v54 native NF4 screen](03_validation/reports/v54_research): large MLP projections were slower; limited narrow-projection opportunity remains unintegrated. No new full-task score.

[v55 measured outcome](CODEX_V55_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[v56 measured outcome](CODEX_V56_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

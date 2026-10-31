# Verified optimization results: v45 through v56

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
| v63 | 83.68 | 48.69 | 4115 | 2.04962 → 2.04963 | 10/10 | FAIL: user limits |

The current <=27-second whole-command target is not met. Historical reports keep their original target. The best observed memory among
these candidates that also satisfy the strict user loss requirement is 4115 MiB (v48/v49), only 20 MiB below v45's 4135 MiB. Their
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

[v54 native NF4 screen](03_validation/reports/v54_research): large MLP projections were slower. Its narrow-projection opportunity was subsequently tested as v56, without a confirmed full-task speed benefit; v56 also failed the user loss requirement.

[v55 measured outcome](CODEX_V55_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[v56 measured outcome](CODEX_V56_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[All five new original ZIPs and checksum index](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v56-20261031). [Selected-source and protected-file audit](04_submission/v53_v56_source_audit.json). All tested new runtime candidates failed the stricter nonincreasing reported-loss rule. Both the active development runtime and public selected solution are restored/verified as exact v45. The <=27-second whole-command target has not been achieved.

[v63 measured outcome](CODEX_V63_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[v57-v62 measured inactive research](CODEX_V57_EXPERIMENT.md): the shared-memory Tensor Core prototypes matched the sampled native results, but stayed slower than native GEMM. v62 passed 36 exact NF4 shape/launch checks. Its launch policy was tested in [v63](CODEX_V63_RESULT.md): 48.69 s / 4115 MiB / original 10/10, with reported eval loss 2.04962 -> 2.04963. No demonstrated whole-command gain or strict nonincreasing-loss result supports replacing v45.

The v57-v63 investigation is complete. Both the active development implementation and the public selected implementation are restored to exact measured v45. The candidate remains available under `experiments/v63`; it is not selected. Historical selected whole-command time is **45.03 s**, peak **4135 MiB**, original **10/10**, reported eval loss **2.04963 -> 2.04960**. Its 26.54-second training loop is only part of the whole command. The <=27-second whole-command target remains **unmet**. Research measurements do not establish a hardware speed limit.

[Original v57-v63 evidence release](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v63-20261031): eight exact research ZIPs inside a bundle, the original full v63 ZIP, and a source/checksum manifest. All three release assets have verified public download links and server SHA-256 digests. [Publication receipt](04_submission/v57_v63_publication.json). [Selected-source and protected-file audit](04_submission/v57_v63_source_audit.json).

# Current selection and latest measurements

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

Public main: selected v45, 45.03 seconds / 4135 MiB / original 10/10.
v48 and v49 are measured memory alternatives (4115 MiB), not replacements.
v46 failed the stricter nonincreasing-loss rule; v47 was unmeasured.
The current whole-command <=27-second target remains unmet.

[Latest decisions and evidence](CODEX_OPTIMIZATION_RESULTS.md).
[Selected result](CODEX_V45_RESULT.md). [Selected source proof](https://github.com/promaladoi/Decentrathon-6.0/blob/main/%D0%9D%D2%B1%D1%80%D0%B6%D1%96%D0%B3%D1%96%D1%82/PUBLICATION_PROOF.json).
Pinned reproducible notebook: `templates/codex_v45.ipynb`.
Development HEAD may contain experimental code; use the exact measured commit.

[Measured v51 result](CODEX_V51_RESULT.md). No confirmed improvement; v45 stays selected.

Same-session v45 solution-only control: 48.55 s / 4135 MiB; v51: 49.03 s.
v52/v52r1 explicit-MMA prototypes matched sampled native values but were slower
and were not integrated. See `03_validation/reports/v52r1_research/README.md`.

[v53/v53r1 verified outcome](CODEX_V53_RESULT.md): not selected; user quality requirement failed.

[v54 native NF4 screen](03_validation/reports/v54_research): large MLP projections were slower. Its narrow-projection opportunity was subsequently tested as v56, without a confirmed full-task speed benefit; v56 also failed the user loss requirement.

[v55 measured outcome](CODEX_V55_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[v56 measured outcome](CODEX_V56_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

[All five new original ZIPs and checksum index](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v56-20261031). [Selected-source and protected-file audit](04_submission/v53_v56_source_audit.json). All tested new runtime candidates failed the stricter nonincreasing reported-loss rule. Both the active development runtime and public selected solution are restored/verified as exact v45. The <=27-second whole-command target has not been achieved.

[v57-v62 measured inactive research](CODEX_V57_EXPERIMENT.md): the shared-memory Tensor Core prototypes matched the sampled native results, but stayed slower than native GEMM. v62 passed 36 exact NF4 shape/launch checks. Its launch policy was tested in [v63](CODEX_V63_RESULT.md): 48.69 s / 4115 MiB / original 10/10, with reported eval loss 2.04962 -> 2.04963. No demonstrated whole-command gain or strict nonincreasing-loss result supports replacing v45.


[v63 measured outcome](CODEX_V63_RESULT.md): not selected; original admission or stricter user requirement failed. The 27-second target remains unmet.

The v57-v63 investigation is complete. Both the active development implementation and the public selected implementation are restored to exact measured v45. The candidate remains available under `experiments/v63`; it is not selected. Historical selected whole-command time is **45.03 s**, peak **4135 MiB**, original **10/10**, reported eval loss **2.04963 -> 2.04960**. Its 26.54-second training loop is only part of the whole command. The <=27-second whole-command target remains **unmet**. Research measurements do not establish a hardware speed limit.

[Original v57-v63 evidence release](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v63-20261031): eight exact research ZIPs inside a bundle, the original full v63 ZIP, and a source/checksum manifest. All three release assets have verified public download links and server SHA-256 digests. [Publication receipt](04_submission/v57_v63_publication.json). [Selected-source and protected-file audit](04_submission/v57_v63_source_audit.json).

# Current status

- Fully archived and source/hash-verified v17r1: `7c5f6d8`, 84.28 / 54.19 s,
  speedup 1.5553x, peak 5601 MiB, eval delta -0.002%, all ten checks.
- Fully archived v18r1: `d01b2d8`, 80.94 / 55.61 s, peak 4701 MiB, 10/10.
- Fully archived v20r1: `22ff279`, 81.23 / 55.81 s, peak 4467 MiB, 10/10.
  This is the lowest collected memory result, not the fastest paired ratio.
- v16 `4a5cfd2` remains the archived fallback: 85.07 / 60.41 s, 5601 MiB, 10/10.
- Initial v17/v18/v20 stopped in a fresh session missing pinned dependencies.
  All three failed source snapshots/logs are preserved; r1 labels preserve retries.
- Kaggle UI control is restored. The serial finite coordinator has been started
  by the assistant; no user action is currently needed to launch these experiments.
- v21 `a67816b` console reports 10/10 at about 55.5 s, 4467 MiB; raw collection pending.
- v22 `7a8b38f`: guarded train-only CLI imports, 7 real-parser CPU tests pass;
  paired T4 measurement running. All CLI callbacks/handler functions are preserved.
- v23 `d8ac022`: bounded process-local reuse of exact frozen MLP weights;
  14 CPU cache tests pass, 4 CUDA cases await validation. It spends extra resident
  memory to test speed. No cross-run cache or new quantization/optimizer is used.
- After v21/v22/v23, a separate CPU/GPU diagnostic profiles an admitted candidate.
  Diagnostic timing is not a contest score. The harness, task and organizer tests
  remain unchanged; all ten rules, including real own-kernel share, still apply.
- User target: 20–30 s total, about 64.7–76.5% below the 85.07 s reference.
  This has not been achieved; memory reduction is a separate metric.

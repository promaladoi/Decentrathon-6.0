# Current status

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

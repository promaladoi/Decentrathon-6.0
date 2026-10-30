# Current status

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

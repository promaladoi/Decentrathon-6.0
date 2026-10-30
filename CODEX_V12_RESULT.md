# v12: skip zero-gradient backward rows

Measured code: `bd2b49ac6914a3b8aa30133b615c1a23ddf34338`.
Two paired T4 runs: base **83.97 / 88.27 s**, solution **74.20 / 74.19 s**.
Medians **86.12 / 74.19 s**, speedup **1.1608x** (**13.9% less runtime**).
Peak memory **14345 / 7385 MiB** (**48.5% less**).
Eval loss **2.04964 / 2.04963**; all **10/10 checks** passed.
**64 kernel tests + 79 Soup tests** passed. Own-kernel share **8.1%**.

The frozen head retains its ordinary in-run fp16 weight cast and skips rows
with an exactly zero gradient. The MLP has the same guarded zero-gradient-row
backward optimization. The forward computation and all per-token metrics remain.
The model, task, dataset, dropout, optimizer, training switches and harness are unchanged.

The first attempted queue did not produce a complete report. A repeat unit-test
run then encountered another live process consuming the GPU. The accepted run
used a fresh `runs_codex_v12_clean` directory with an idle GPU and completed all
four training runs and both evaluations. A notebook export initially referred to
the old output path; the reports were subsequently checked and exported from the
correct directory, without repeating or editing the measured results.

Raw byte-preserved reports: `03_validation/reports/codex_v12/`.
The archive's source, result SHA-256 and solution SHA-256 were verified locally.
This is the strongest completed v1-v12 result. The newer v13 combines Vladimir's
unpadded model path with Nurzhigit/Codex parallel vocabulary reductions and is
an independent experiment pending a full benchmark.

# v37 measured: admitted, no demonstrated full-command improvement

Exact source `59715e5b0bd45b2ed27fb1cb2b3a1480cc5fe8c6`.
Two paired base/solution T4 runs; all 53 solution source files and the original
result/check checksums were verified from the downloaded raw archive.

| Version | Paired base s | Solution s | Every-run peak MiB | Eval delta | Checks |
|---|---:|---:|---:|---:|---:|
| v34 reference | 78.68 | 43.91 | 7151 | -0.001% | 10/10 |
| v37 | 79.54 | 44.64 | 7151 | -0.002% | 10/10 |

v37 solution runs: 44.17 / 45.12 seconds. Training-loop median 24.18 seconds.
327 solution tests and 79 Soup regression tests passed. Own kernels: at least
7.7% of the timed training steps. Same task, tokens, metrics, adapter and rewind.
The reported evaluation loss was 2.04963 vs base 2.04968.

The additional `user_constraints.json` checks the **maximum of every solution
run**, not just a summary peak. Memory and reported evaluation loss meet the
user limits; the **30-second total-time target is not met**. v34 remains selected.

The separate fresh v34 diagnostic is not a new admitted paired score. GEMM
kernels account for about 69.47% of active profiled GPU kernel time. Background
compilation finishes before step one. The metadata inference optimization was
confirmed active; the full initial package scan happens before solution.apply
in the unchanged harness. No bypass of that startup or measurement is introduced.

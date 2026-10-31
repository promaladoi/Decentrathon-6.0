# v57-v62 inactive research, selected v45 preserved

No prototype here is installed in solution.apply. The selected runtime is the
exact measured v45. No complete-task speedup or <=27-second result is claimed.

The first two v57 revisions stopped at compilation. Exact failed sources and
errors were retained. The corrected v57r2 then passed eight numerical cases.
Subsequent prefetch, ldmatrix, address and warp/tile changes remained numerically
correct on the sampled cases, but slower than native GEMM.

| Screen | Numerical cases | Representative M=512 prototype/native |
|---|---:|---:|
| v57r2 | 8 | 1.682 / 0.748 ms |
| v58 | 8 | 1.469 / 0.751 ms |
| v59 | 16 | 1.226 / 0.711 ms |
| v60 | 12 | 1.289 / 0.749 ms |

These are separate microbenchmarks with their own paired native timings, not
training measurements or a comparable cross-screen hardware clock. None beats
its native control. All unmodified samples, exact source commits, and raw ZIPs
are in the respective `03_validation/reports/v*_research` folders.

v61 removes most Python launch gaps only for an isolated diagnostic by using
warmed graph replay for BOTH implementations. GPU time remains worse: at M=512,
1.101 vs 0.644 ms; at M=832, 1.715 vs 1.150 ms. Graphs were not added to training
or used to bypass any original observer. This rules out launch overhead as the
main explanation for the current dense-MMA prototype's gap.

v62 now screens the unchanged exact v45 NF4 decoder across block/warp geometries.
Every screen output must match native bitsandbytes exactly. Production remains
unchanged until a worthwhile launch choice passes the full original task,
regressions, <=7151 MiB and nonincreasing reported eval loss.

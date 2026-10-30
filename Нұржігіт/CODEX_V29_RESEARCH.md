# v29 research: fuse the MLP's two backward matrix products and gradient sum

Inactive research kernel, not selected for training. The v17 diagnostic's
largest remaining GPU category was GEMM. `fused_mlp_gradient.py` computes two
independent FP16 tensor-core matrix products with FP32 accumulators, rounds each
to FP16 as native matmul does, and sums their casts in FP32. This avoids separate
FP16 output buffers, FP32 cast kernels and the final addition launch. It is an
independent matrix-product fusion; organizer SwiGLU stays unchanged/excluded.

Six tile configurations must first be compared against the equivalent native
two-matmul path on T4 outside contest measurements. If the diagnostic is slower,
this research stays inactive. An apparent standalone win would still require
whole-model gradient tests and a new original paired 10/10 benchmark before any
claimed improvement. No compiler artifact is carried into a measured run.

The first T4 test reported 12 passed / 4 failed: two of 1,279,488 entries in the
large shape differed by 0.00390625, while the global relative-L2 <0.001 check
passed. The fixed 0.003 absolute tolerance on a cancellation result was not a
valid bound for two independently FP16-rounded products. The revised test keeps
the global L2 bound and checks two ulps per native partial product, plus an exact
constructed cancellation case that rejects summing before FP16 rounding.
The token count is now runtime, avoiding a new compiled variant for every batch
length. No training or performance claim follows from this test correction.

Prepared v30/v31 include the earlier inactive research tests; their queued
executions were aborted after this research-cell error. They were never admitted.
New exact snapshots are required before running the corrected full suite.

The revised per-entry oracle also failed (8 cases in the full suite). The kernel
remains rejected for production use; no standalone speed claim is available.
Implementation, strict failing tests and microbenchmark are kept together under
`02_kernel_research/`, not imported by `solution.apply()`. v32/v33 stopped before
training. Production candidates continue using native PyTorch GEMM. This is a
separation of unused research, not removal of an active-kernel regression test.

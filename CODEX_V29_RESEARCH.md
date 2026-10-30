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

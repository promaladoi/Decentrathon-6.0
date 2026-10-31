# v52 inactive explicit-MMA screen

Generated PTX contains the T4 m16n8k8 Tensor Core instruction. Six small mapping tests and two representative FP16 GEMMs match native FP16 outputs exactly in this screen. This does not prove arbitrary inputs, backward correctness or full-task quality.

The prototype is substantially slower than native GEMM, including after aligned loads and K-loop scheduling in v52r1. It is not installed in solution.apply. There is no new training result, 10/10 claim, or <=27-second claim.

[NVIDIA fragment specification](https://docs.nvidia.com/cuda/archive/11.0_GA/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688). The team wrote the kernel; no third-party kernel implementation is included.

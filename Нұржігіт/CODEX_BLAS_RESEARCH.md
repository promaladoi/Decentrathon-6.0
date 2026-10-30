# Vendor BLAS selection research

No production backend flag is changed. After all paired jobs, compare the
current default with PyTorch's `preferred_blas_library('cublaslt')` on real
MLP/head matrix shapes in a separate T4 process. Both are ordinary PyTorch
vendor backends, with unchanged dtypes and precision flags. The original
preference is restored and the process exits before any later measurement.

This is a microbenchmark, not an admitted speedup. A useful standalone result
would still need its own clean full paired benchmark, quality and all ten rules.
If it is slower or unsupported, the production default remains untouched.

API checked against the local pinned Torch 2.11 code and official documentation:
https://docs.pytorch.org/docs/2.11/backends.html#torch.backends.cuda.preferred_blas_library

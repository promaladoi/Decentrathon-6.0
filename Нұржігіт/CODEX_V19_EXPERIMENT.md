# v19: keep the native dropout mask directly (GPU pending)

Based on v18, this removes two full-shape operations from compact LoRA dropout:
filling the dummy input with ones and extracting another boolean mask with
`ne(0)`. The same ATen CUDA native dropout call draws a full padded mask. Its
numeric dummy output is discarded; only the mask is used on the real tokens.
The input values do not affect this CUDA mask or its RNG consumption.

This revisits v14's mask producer **after** v16 fixed the actual strided-index
bug. The old failed v14/v15 evidence remains preserved. Their earlier failure
does not prove that native and functional masks differ. Conversely, reviewing
the source does not replace the required GPU parity tests for this revision.

Primary implementation reviewed for the installed PyTorch version:
[functional dropout dispatch](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp)
and [CUDA mask generation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu).
The guard still restricts this path to contiguous CUDA fp16/fp32 activations and
ordinary training dropout with 0 < p < 1. No RNG formula or probability changes.

Six additional GPU cases compare **every full-mask bit and post-call RNG state**
for both storage types and three sequence lengths. The existing eight exact
output, gradient, successive-call RNG and noncontiguous-index cases remain.
Thus the dropout preflight contains 14 cases; the full solution suite has 164.
These tests must pass before the unchanged two-pair contest measurement runs.

No CUDA tests, speed, quality or 10/10 result are claimed for v19 yet. The stored
v16 archive remains the fully verified candidate. Future results must identify
this exact commit and pass source/hash checks before replacing it.

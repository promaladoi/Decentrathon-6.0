# v64: shared-RMS rematerialization of LoRA-A inputs

Starting implementation: exact measured v45, commit `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`.

The guarded private packed path reconstructs the LoRA-A backward input from
the RMS producer's already-required FP32 residual, inverse RMS and frozen
norm weight, plus the same bit-packed native dropout mask. It saves no
second large normalized or dropped tensor. The reconstruction uses one own
elementwise Triton kernel and adds no GEMM or random draw.

Native full padded dropout RNG, FP32 multiply ordering, the FP16 LoRA-A
matrix multiplication, both native backward GEMMs and FP32 gradient joins
are preserved. Ordinary model calls, small shapes, hooks, overridden Linear
or Dropout forwards, frozen adapters and unsupported autocast cases retain
the existing operation. The public RMSNorm API still returns two outputs;
the private producer additionally exposes its existing nondifferentiable
inverse RMS using the same forward/backward kernels.

This is an unmeasured candidate. No speed, peak-memory, quality or original
ten-check success is claimed before the original full paired measurement.
The selected public implementation remains v45. The 27-second whole-command
target has not been achieved.

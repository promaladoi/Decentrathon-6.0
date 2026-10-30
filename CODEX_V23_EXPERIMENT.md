# v23: spend part of the saved VRAM on frozen MLP weight reuse (GPU pending)

The explicit priority is the same complete training command in 20–30 seconds.
This revision, built on v22, tests a time/memory tradeoff: decode a frozen NF4
MLP matrix once inside this run and retain its exact fp16 values for subsequent
forward/backward calls. Packed parameters, quantization settings, precision,
LoRA, optimizer, random draws and training outputs remain unchanged.

The cache is CUDA fp16 only, for supported frozen NF4 matrices with at least
4194304 elements, and is capped at 2400000000 resident bytes. The current model's
84 MLP matrices would occupy about 2205 MiB. That is extra residency, not a peak
memory measurement. It may worsen the memory score while reducing runtime;
the unchanged harness must establish whether this tradeoff is useful.

No model/adapter/decoded weight is loaded from a preceding run or disk cache.
The cache is new in every measured process; initial decoding/allocation remains
inside the timed command. Original quantized values are decoded with the same
decoder, so repeated use does not approximate them or skip their matrix products.

Entries track owner identity, tensor pointers/strides/dtypes and PyTorch mutation
versions of packed weights and quantization tensors. Mutations/replacements,
layout or dtype changes invalidate the entry. Weak ownership releases resident
storage when the parameter dies. Trainable tensors, unversioned inference tensors,
aliasing decoder outputs and exhausted budgets keep the transient decoder path.
As with normal PyTorch caches, unsupported direct `.data` mutation bypassing
version tracking must not be used; this task's base weights remain frozen.

CPU cache-contract tests cover exact reuse, mutations of weights/scales/offset,
replaced scale tensors, nested states, dtype/budget handling, owner deletion and
fallbacks. Four CUDA tests check the actual NF4 decoded values, reuse and
invalidation on large gate/down shapes. Training quality, memory, speed and all
ten checks remain pending. If removal of repeated decodes drops actual own-kernel
share below 5%, this candidate is rejected by the unchanged rule check.

# v69: reduce retained activations and reuse only small frozen weights

Candidate, not a measured improvement. Starting math and memory path: v64r2
(`5c36601c8302da8e0910b7697637fea4b3a0393d`). LoRA-A inputs are reconstructed
from the existing shared RMS residual and bit-packed native dropout mask.

Internal frozen NF4 matrices are reused only when they have at most 524288
elements. On the fixed Qwen model this admits the small K/V matrices, while
Q/O and MLP matrices stay transient. The cache is capped at 67108864 bytes.
Its exact decoder, mutation, layout, owner, stream and output guards derive
from the already qualified v65-v68 experiment. Public decoding stays fresh.

All work starts inside each fresh measured subprocess. The native CLI,
512-row vocabulary chunk, FP32 rounding boundaries, dropout random sequence,
model, optimizer, data and 64 steps stay unchanged. Team credits stay in the
original module headers. The original 15 protected files must match their
recorded hashes. Qualification requires all native tests and original 10/10,
every solution run <=7151 MiB and nonincreasing reported evaluation loss.

Select only after comparing complete command time and memory with exact v45.
No 25-second result is claimed before a completed measurement.

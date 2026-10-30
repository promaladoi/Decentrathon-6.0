# Current experiment

Code e19b464 (v10) preserves half logits at the Accelerate output boundary and
uses float32 loss/entropy reductions. GPU tests and the paired benchmark are pending.
This is not yet an accepted improvement. The pinned candidate notebook still
runs validated v9 (57182a2, 10/10 checks, 1.1075x speedup).

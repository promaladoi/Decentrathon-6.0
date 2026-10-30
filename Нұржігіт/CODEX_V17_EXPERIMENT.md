# v17: paired-byte NF4 decoding experiment

Code: `2c2baf4fedcaff288017955b26069d8947ca9b22`; validation runner/microbenchmark
snapshot: `7c5f6d83a5499886d749128832925e27bf1533d8`.
This builds on v16 and changes only NF4 weight reconstruction and its startup
compile signature. Matmuls stay in the same cuBLAS path. No dense weights are
cached and no task/model/optimizer/dropout setting is changed.

The old decoder assigns one lane per output weight, rereading each packed byte
and rebuilding its scale for both nibbles. The new decoder assigns one lane per
byte, shares the scale, then joins high/low values into contiguous output stores.
Odd tails and the nested scale format are handled explicitly. Fallback remains
for unsupported cases, and all three shared decoder references are patched.
Vladimir's shared format/decoder is the reference and remains credited.

The v16 run's own timing table records 2321.832 ms of NF4 decoding in its first
solution run (timed steps 3+). This identifies a target; it is not a predicted
end-to-end gain. Removing this operation entirely would still not approach a
70–90% total-time reduction by itself. Broader profiling is still needed.

Local CPU verification executes the actual kernel AST through a NumPy pointer
model against an independent unpackbits/repeat oracle: **104 cases passed**.
These cover all byte/nibble values, odd tails, block boundaries and nested-scale
boundaries; masked loads/stores are checked for bounds. Dyadic values isolate
address/layout correctness. **This does not test Triton GPU code generation,
GPU floating point parity, speed, gradients or training quality.**

There are **23 new GPU cases** (138 total kernel tests): bitwise output equality
to the old decoder for synthetic/real quantized weights, frozen-linear output
and input-gradient equality, and the production warmup signature. Tolerances are
zero for the new parity tests. The ordinary task and all ten checks still apply.

`03_validation/benchmark_nf4_pair.py` separately compares old/new GPU-event and
wall latencies over four Qwen matrix shapes, with alternating order. It is not a
training score. The contest harness uses its own fresh processes/cache as usual.

Layout semantics reference: [Triton join](https://triton-lang.org/main/python-api/generated/triton.language.join.html).
The kernel implementation is team-written, not copied from that documentation.

**v17 remains unverified on T4.** The accepted candidate stays v16 until its
unit tests, two paired runs, quality checks and raw hashes are verified.

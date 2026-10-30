# v20: save compact dropout mask bits (GPU pending)

Built on v19's direct native mask and v18's checked tied-weight storage. The
forward Triton kernel applies dropout to real-token activations and packs those
same selected mask bits, eight per byte, in the same pass. Autograd saves only
the compact byte tensor. Backward reads its bits directly; it no longer retains
the padded boolean mask or row indices for every LoRA branch of every layer.

Stored mask size becomes `ceil(real_tokens * hidden / 8)` bytes instead of
`batch * padded_length * hidden` bytes. The full padded random draw still occurs
exactly once per dropout call. No padded draw is skipped and no RNG is replaced.
This targets retained activation memory and backward memory traffic. Overall
peak memory and runtime improvement must be measured; none is claimed yet.

Ten new GPU cases cover all 256 byte patterns, partial final bytes, output guards,
odd feature counts, exact output/gradient/RNG behavior and noncontiguous row
indices. The complete dropout preflight has 24 cases, the solution suite 174.
All tests precede the ordinary unchanged two-pair T4 harness and ten rule checks.

`03_validation/verify_packed_dropout_layout.py` separately evaluates the actual
kernel bodies through a bounds-checked CPU pointer model against an independent
NumPy packbits/unpackbits oracle. This validates address/bit layout, not CUDA
compilation, CUDA floating-point behavior, RNG or training quality/performance.

The bounded coordinator queues v17, v18 and v20, using exact source commits in
isolated serial workers. v19 is the separately committed intermediate source,
not an independently measured result. v16 remains the fully verified fallback.

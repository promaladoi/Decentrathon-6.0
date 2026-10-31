# v55: recompute one frozen MLP activation

Starting runtime: exact selected v45, commit
`7a3db8fd5e24bbcb58affb475007a5cf93c56f07`. No v53 dispatch change is enabled.

Instead of retaining both [tokens, 8960] FP16 gate/up activations, save gate
and the [tokens, 1536] FP16 input. Recreate up with the same frozen NF4
dispatcher, input, shape and rounding in backward. No RNG operation is added.
Frozen up-weight/quantization mutations are rejected before recomputation;
retained graphs and a caller's subsequent half-input mutation are tested.

This trades one additional projection per layer backward for lower saved
activation memory. It may cost time. No speed, memory, loss or original
10/10 result is claimed before complete measurement. The current <=27-second
whole-command target and <=7151 MiB bound remain unchanged.

Ten targeted GPU tests require bitwise forward/input-gradient agreement,
unchanged RNG, retained-graph agreement and smaller saved activations. Existing
team and original Soup tests then run, followed by the untouched original
paired harness/checks and stricter user limits. Any failed stage is archived.

Public selected solution remains measured v45 until a justified selection.

# v30: avoid materializing an unused dense attention mask

Experimental on top of the pending v28 path. v29's matrix-product research stays
inactive. Native variable-length attention uses the validated cumulative lengths
and its causal flag rather than the shared dense mask. v30 avoids creating that
dense mask only when a valid variable-length plan exists, the supported head
dimension is used, and every layer is causal. Fallbacks still build the original
mask. Positions, input rows, random draws and the logical attention are unchanged.

The edit to Vladimir's shared `unpad.py` is Nurzhigit's integration change, with
the shared implementation's attribution preserved. A CPU guard test and the
existing native-vs-padded attention gradients, isolation and tiny-Qwen tests
cover the preconditions; full training still requires the original 10/10 rules.

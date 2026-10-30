# v24: variable-length causal attention without padded queries (GPU pending)

On top of v23, this changes attention storage/execution while preserving the
logical batch and every sequence boundary. The current shared path removes pads
from projections/MLPs but pads q/k/v again for SDPA. v24 keeps real tokens and
passes cumulative per-sequence lengths to PyTorch's native efficient-attention
operator, with its registered autograd implementation and causal mask type.

The exact operator call is the efficient backend used by
[PyTorch 2.11 jagged SDPA](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nested/_internal/sdpa.py).
[Nested-tensor documentation](https://docs.pytorch.org/docs/2.11/nested.html)
provides the variable-length context. No external attention package is imported;
the Torch/CUTLASS backend is the same framework backend available to ordinary
SDPA. The new dispatch is restricted to the inspected Torch 2.11 / T4 SM75 case.

Vladimir's existing RoPE kernel writes compact storage and still uses the cos/sin
of each token's original position. Cumulative lengths delimit the original batch
sequences; no token can attend to another example or a future token. q/k/v LoRA
branches retain their original order, full padded dropout shape and RNG draws.
The loss, entropy, predictions, adapter, optimizer and Soup settings stay intact.
An empty sequence or unsupported dimension keeps the original attention path.

Four CPU plan tests check offsets/batch boundaries, including gaps/left padding
and empty-sequence fallback. Eleven CUDA cases check native outputs/input
gradients, contiguous and head-major strides, causal/batch isolation, full model
loss/LoRA gradients/entropy and the exact dropout RNG state. Full suite: 214 cases.
Those CUDA cases and all ten contest rules must pass before admission.

v23 has now completed 10/10: 80.68 / 52.76 seconds, 6747 MiB, eval delta -0.003%.
It traded memory for a modest runtime saving, and the 20–30-second target remains
unmet. v24 is a separate unmeasured experiment; no predicted speedup is claimed.

"""Move redundant dtype conversions around the fixed frozen fp16-compute MLP.

Experimental; install after the existing Qwen MLP patch. This reuses the
organizer's unchanged SwiGLU kernels, which do not count as original kernels.
See compressed_swiglu._eligible for the strict producer and fixed-model guards.

Original eligible path, with h=float16 and f=float32:
 gate = widen_f(gemm_h(cast_h(x))); up = widen_f(gemm_h(cast_h(x)))
 y = widen_f(gemm_h(cast_h(swiglu_f(gate, up))))

The gate/up widened values are exactly representable in h. The organizer's
SwiGLU kernel explicitly computes in f, independent of storage dtype. Writing
its output directly to h performs the same rounding as the original cast_h.
The resulting MLP output still has dtype f. In backward the intermediate
gradients encounter the same half roundings, in the same order.
"""

import functools

import torch

from .compressed_swiglu import _eligible
from .swiglu import swiglu


def patch_qwen2_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    original = Qwen2MLP.forward
    if getattr(original, "_soup_narrow_mlp", False):
        return

    @functools.wraps(original)
    def forward(self, x):
        if not _eligible(self, x):
            return original(self, x)
        # Keep two distinct cast nodes: sharing one x_half would accumulate
        # the two projection gradients in fp16 instead of the original fp32.
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        return self.down_proj(swiglu(gate, up)).to(x.dtype)

    forward._soup_narrow_mlp = True
    Qwen2MLP.forward = forward

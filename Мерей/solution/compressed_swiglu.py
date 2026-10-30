"""Lossless saved-activation compression for the fixed frozen NF4 Qwen MLP.

Experimental: copy into solution/ and call patch_qwen2_mlp() AFTER the existing
solution patches. This reuses the organizer's SwiGLU kernels, which do not count
as original kernels. It changes only how their two saved activations are stored.

bitsandbytes 0.50.2 Linear4bit, with an explicitly set fp16 compute dtype, first
computes an fp16 result and then converts it to its input dtype. Its fp32 output
therefore contains exactly representable fp16 values. This premise does NOT hold
for PEFT's Linear4bit wrapper, which adds its adapter result after that conversion.
The integration below accepts only the plain frozen bitsandbytes layer type.
"""

import functools

import torch
import triton
from bitsandbytes.nn import Linear4bit as _Linear4bit

from .swiglu import (BLOCK, example_swiglu_bwd_kernel,
                     example_swiglu_fwd_kernel)

_BNB_FORWARD = _Linear4bit.forward


class CompressedSwiGLU(torch.autograd.Function):
    """Only call with gate/up proven to be widened fp16 projection results."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        # The forward receives the same fp32 values and uses the unchanged
        # organizer kernel. Only the persistent backward storage becomes fp16.
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate.to(torch.float16), up.to(torch.float16))
        ctx.input_dtype = gate.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        gate_half, up_half = ctx.saved_tensors
        # Preserve the original fp32 gradient arithmetic and stores. Allocating
        # gradients with empty_like(gate_half) here would change the contract.
        grad = grad.to(ctx.input_dtype).contiguous()
        dgate = torch.empty(gate_half.shape, device=gate_half.device,
                            dtype=ctx.input_dtype)
        dup = torch.empty_like(dgate)
        n = gate_half.numel()
        # The organizer kernel explicitly casts every load to tl.float32.
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate_half, up_half, dgate, dup, n, BLOCK=BLOCK)
        return dgate, dup


def _eligible(mlp, x):
    """Reject adapter wrappers, alternate compute dtypes and trainable weights."""
    if (not isinstance(x, torch.Tensor) or not x.is_cuda
            or x.dtype != torch.float32 or x.numel() == 0
            or getattr(getattr(mlp, "config", None), "hidden_act", None) != "silu"):
        return False
    import bitsandbytes as bnb

    # Dtype movement must not change values exposed to forward observers, or
    # bypass hooks that modify projection results. Reject global hooks too.
    module_impl = torch.nn.modules.module
    if any(getattr(module_impl, name, None) for name in (
            "_global_forward_pre_hooks", "_global_forward_hooks",
            "_global_backward_pre_hooks", "_global_backward_hooks")):
        return False
    if (getattr(_BNB_FORWARD, "__module__", None) != "bitsandbytes.nn.modules"
            or getattr(_BNB_FORWARD, "__qualname__", None) != "Linear4bit.forward"
            or hasattr(_BNB_FORWARD, "__wrapped__")
            or bnb.nn.Linear4bit.forward is not _BNB_FORWARD):
        return False
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(mlp, name, None)
        if (type(layer) is not bnb.nn.Linear4bit
                or layer.compute_dtype != torch.float16
                or layer.compute_type_is_set is not True
                or layer.bias is not None
                or any(p.requires_grad for p in layer.parameters())):
            return False
        if (getattr(layer.forward, "__func__", None) is not _BNB_FORWARD
                or any(getattr(layer, key, None) for key in (
                    "_forward_pre_hooks", "_forward_hooks",
                    "_backward_pre_hooks", "_backward_hooks"))):
            return False
        weight = layer.weight
        state = getattr(weight, "quant_state", None)
        if (type(weight) is not bnb.nn.Params4bit or weight.device != x.device
                or weight.dtype != torch.uint8 or state is None
                or state.quant_type != "nf4"):
            return False
    return True


def patch_qwen2_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    original = Qwen2MLP.forward
    if getattr(original, "_soup_compressed_swiglu", False):
        return

    @functools.wraps(original)
    def forward(self, x):
        if not _eligible(self, x):
            return original(self, x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # The exact plain layer type and compute dtype establish losslessness.
        return self.down_proj(CompressedSwiGLU.apply(gate, up))

    forward._soup_compressed_swiglu = True
    Qwen2MLP.forward = forward

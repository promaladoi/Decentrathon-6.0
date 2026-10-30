"""Qwen2's MLP on 4-bit weights with fp16 activations, forward and backward.

bitsandbytes' Linear4bit returns its result in the dtype of its input. Here the MLP input is fp32
(k-bit preparation keeps the norms in fp32), so in transformers gate and up come back as fp32:
2 x tokens x 8960 values per layer that the SwiGLU keeps for backward, written once as fp32 and
cast back to fp16 for the next matmul. The values themselves are fp16 results of fp16 matmuls, so
the fp32 copies hold nothing extra. Here the input is cast to fp16 once (Linear4bit casts it once
per layer), gate, up and their product stay fp16, and backward adds the two input gradients in
fp32 exactly as autograd adds the two casts' gradients. Every value equals transformers' result
with Linear4bit, while the saved activations of the MLP take half the memory and the fp32 casts
around the three matmuls disappear.

The weights are unpacked by our NF4 kernel (nf4.py) right before each matmul and dropped after it,
as bitsandbytes does. silu(gate) * up runs on the organizers' example SwiGLU kernels, unchanged;
they are theirs and do not count as ours.
"""

import torch
import triton

from .nf4 import dequantize_nf4, supported as nf4_supported
from .swiglu import BLOCK as SWIGLU_BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel


def _weight(layer, dtype):
    return dequantize_nf4(layer.weight, layer.weight.quant_state, dtype)


class MLP4bit(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for three bias-free bitsandbytes Linear4bit layers."""

    @staticmethod
    def forward(ctx, x, gate, up, down, dtype):
        x16 = x.to(dtype)
        g = torch.nn.functional.linear(x16, _weight(gate, dtype))
        u = torch.nn.functional.linear(x16, _weight(up, dtype))
        del x16
        h = torch.empty_like(g)
        n = g.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](g, u, h, n, BLOCK=SWIGLU_BLOCK)
        out = torch.nn.functional.linear(h, _weight(down, dtype))
        ctx.layers, ctx.dtype, ctx.x_dtype = (gate, up, down), dtype, x.dtype
        # kept as attributes, not saved tensors, so backward can free them as soon as it is done
        ctx.gu = (g, u) if ctx.needs_input_grad[0] else None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate, up, down = ctx.layers
        dtype = ctx.dtype
        g, u = ctx.gu
        ctx.gu = None
        # MatMul4Bit.backward: grad @ dequantized weight, in the gradient's dtype
        dh = torch.matmul(grad_out.to(dtype), _weight(down, dtype)).contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](dh, g, u, dg, du, n,
                                                                   BLOCK=SWIGLU_BLOCK)
        del dh, g, u
        dx = torch.matmul(dg, _weight(gate, dtype)).float()
        del dg
        dx += torch.matmul(du, _weight(up, dtype)).float()  # autograd sums the two casts in fp32
        return dx.to(ctx.x_dtype), None, None, None, None


def _compute_dtype(layers):
    """fp16/fp32 compute dtype shared by three plain NF4 Linear4bit layers, else None."""
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import modules as bnb_modules
    except ImportError:
        return None
    dtype = None
    for layer in layers:
        if not isinstance(layer, bnb.nn.Linear4bit) or layer.bias is not None:
            return None
        fix = getattr(bnb_modules, "fix_4bit_weight_quant_state_from_module", None)
        if fix is not None:
            fix(layer)
        state = getattr(layer.weight, "quant_state", None)
        if not layer.weight.is_cuda or not nf4_supported(state):
            return None
        if layer.compute_dtype not in (torch.float16, torch.float32) or \
                (dtype is not None and layer.compute_dtype != dtype):
            return None
        dtype = layer.compute_dtype
    return dtype


def make_mlp_forward(previous):
    def forward(self, x):
        if not x.is_cuda or getattr(getattr(self, "config", None), "hidden_act", "silu") != "silu":
            return previous(self, x)
        dtype = _compute_dtype((self.gate_proj, self.up_proj, self.down_proj))
        if dtype is None:
            return previous(self, x)
        return MLP4bit.apply(x, self.gate_proj, self.up_proj, self.down_proj, dtype)

    return forward


def patch_qwen2_mlp_4bit():
    """Class patch on top of whatever Qwen2MLP.forward is (the example's), which stays the
    fallback for layers that are not plain NF4 Linear4bit."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    cls.forward = make_mlp_forward(cls.forward)


def warmup():
    """Compile the fp16 variants of the example SwiGLU kernels this MLP launches; see warmup.py."""
    f16 = torch.float16
    n = 16 * 8960  # tokens x 8960 is always a multiple of 16
    example_swiglu_fwd_kernel.warmup(f16, f16, f16, n, grid=(1,), BLOCK=SWIGLU_BLOCK)
    example_swiglu_bwd_kernel.warmup(f16, f16, f16, f16, f16, n, grid=(1,), BLOCK=SWIGLU_BLOCK)

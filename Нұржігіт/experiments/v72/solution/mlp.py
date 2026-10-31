"""Qwen2's MLP on 4-bit weights with fp16 activations, forward and backward.

Team integration originally by Vladimir, commit
987da36e1efa90652e1a7f8c48afef50685dbc40. Merey's integration preserves the
public MLP output dtype, retained-graph reuse and narrow frozen-layer guards.

bitsandbytes' Linear4bit returns its result in the dtype of its input. Here the MLP input is fp32
(k-bit preparation keeps the norms in fp32), so in transformers gate and up come back as fp32:
2 x tokens x 8960 values per layer that the SwiGLU keeps for backward, written once as fp32 and
cast back to fp16 for the next matmul. The values themselves are fp16 results of fp16 matmuls, so
the fp32 copies hold nothing extra. Here the input is cast to fp16 once (Linear4bit casts it once
per layer), gate, up and their product stay fp16, and backward adds the two input gradients in
fp32 exactly as autograd adds the two casts' gradients. Every value equals transformers' result
with Linear4bit, while the saved activations of the MLP take half the memory and the fp32 casts
around the three matmuls disappear.

Our NF4 kernel (nf4.py) preserves exact decoded weights. Internal read-only matmuls may reuse
them within the enabled per-process byte budget; overflow uses fresh decode. Public decoder
calls still return fresh tensors. silu(gate) * up uses the unchanged organizers' example kernels;
they are theirs and do not count as ours.
"""

import torch
import triton

from .nf4 import _linear_forward, dequantize_nf4, supported as nf4_supported, _readonly_weight
from .swiglu import BLOCK as SWIGLU_BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel
from .backward_sums import sum_half_gradients


def _weight(layer, dtype):
    return _readonly_weight(layer.weight, layer.weight.quant_state, dtype)


class MLP4bit(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for three bias-free bitsandbytes Linear4bit layers."""

    @staticmethod
    def forward(ctx, x, gate, up, down, dtype):
        x16 = x.to(dtype)
        g = _linear_forward(x16, gate.weight, None, gate.weight.quant_state)
        u = _linear_forward(x16, up.weight, None, up.weight.quant_state)
        del x16
        h = torch.empty_like(g)
        n = g.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](g, u, h, n, BLOCK=SWIGLU_BLOCK)
        out = _linear_forward(h, down.weight, None, down.weight.quant_state)
        ctx.layers, ctx.dtype, ctx.x_dtype = (gate, up, down), dtype, x.dtype
        # Saved tensors preserve version checking, saved-tensor hooks and a
        # second backward when the caller explicitly retains this graph.
        ctx.save_for_backward(g, u)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate, up, down = ctx.layers
        dtype = ctx.dtype
        g, u = ctx.saved_tensors
        # MatMul4Bit.backward: grad @ dequantized weight, in the gradient's dtype
        dh = torch.matmul(grad_out.to(dtype), _weight(down, dtype)).contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](dh, g, u, dg, du, n,
                                                                   BLOCK=SWIGLU_BLOCK)
        del dh, g, u
        gate_dx = torch.matmul(dg, _weight(gate, dtype))
        del dg
        up_dx = torch.matmul(du, _weight(up, dtype))
        del du
        dx = sum_half_gradients((gate_dx, up_dx))
        del gate_dx, up_dx
        return dx.to(ctx.x_dtype), None, None, None, None


def _compute_dtype(layers):
    """The fixed frozen plain NF4/fp16-compute MLP, otherwise eager fallback."""
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import modules as bnb_modules
    except ImportError:
        return None
    module_impl = torch.nn.modules.module
    if any(getattr(module_impl, name, None) for name in (
            '_global_forward_hooks', '_global_forward_pre_hooks',
            '_global_backward_hooks', '_global_backward_pre_hooks')):
        return None
    native_forward = bnb.nn.Linear4bit.forward
    if (getattr(native_forward, '__module__', None) != 'bitsandbytes.nn.modules'
            or getattr(native_forward, '__qualname__', None) != 'Linear4bit.forward'
            or hasattr(native_forward, '__wrapped__')):
        return None
    dtype = None
    for layer in layers:
        if (type(layer) is not bnb.nn.Linear4bit or layer.bias is not None
                or any(p.requires_grad for p in layer.parameters())
                or getattr(layer.forward, '__func__', None) is not native_forward
                or any(getattr(layer, name, None) for name in (
                    '_forward_hooks', '_forward_pre_hooks',
                    '_backward_hooks', '_backward_pre_hooks'))
                or hasattr(layer, '_hf_hook')):
            return None
        fix = getattr(bnb_modules, "fix_4bit_weight_quant_state_from_module", None)
        if fix is not None:
            fix(layer)
        state = getattr(layer.weight, "quant_state", None)
        if (not layer.weight.is_cuda or layer.weight.dtype != torch.uint8
                or not nf4_supported(state) or state.dtype != torch.float16
                or layer.weight.device != state.code.device):
            return None
        if layer.compute_dtype != torch.float16 or not layer.compute_type_is_set or \
                (dtype is not None and layer.compute_dtype != dtype):
            return None
        dtype = layer.compute_dtype
    return dtype


def make_mlp_forward(previous):
    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.numel() == 0
                or x.dtype not in (torch.float16, torch.float32)
                or getattr(getattr(self, "config", None), "hidden_act", "silu") != "silu"):
            return previous(self, x)
        dtype = _compute_dtype((self.gate_proj, self.up_proj, self.down_proj))
        if dtype is None:
            return previous(self, x)
        if (x.device != self.gate_proj.weight.device
                or any(x.device != layer.weight.device for layer in
                       (self.up_proj, self.down_proj))
                or x.shape[-1] != self.gate_proj.in_features
                or x.shape[-1] != self.up_proj.in_features):
            return previous(self, x)
        return MLP4bit.apply(x, self.gate_proj, self.up_proj, self.down_proj, dtype).to(x.dtype)

    return forward


def patch_qwen2_mlp_4bit():
    """Class patch on top of whatever Qwen2MLP.forward is (the example's), which stays the
    fallback for layers that are not plain NF4 Linear4bit."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    cls.forward = make_mlp_forward(cls.forward)


def warmup(parts=("fwd", "bwd")):
    """Compile the fp16 variants of the example SwiGLU kernels this MLP launches; see warmup.py."""
    from .warmup import load_compiled

    f16 = torch.float16
    n = 16 * 8960  # tokens x 8960 is always a multiple of 16
    if "fwd" in parts:
        load_compiled(example_swiglu_fwd_kernel.warmup(f16, f16, f16, n, grid=(1,), BLOCK=SWIGLU_BLOCK))
    if "bwd" in parts:
        from .backward_sums import warmup as warmup_backward_sums
        warmup_backward_sums()
        load_compiled(example_swiglu_bwd_kernel.warmup(f16, f16, f16, f16, f16, n, grid=(1,),
                                         BLOCK=SWIGLU_BLOCK))

"""Frozen MLP backward over nonzero-gradient rows, with unchanged full forward.

This is algebraic zero-work removal, not a different batch, loss or dropout.
The retained organizer SwiGLU backward is explicitly reused and remains excluded
from original-kernel accounting. Gate/up input gradients still sum in fp32.
"""
import torch
import triton

from .swiglu import swiglu, example_swiglu_bwd_kernel, BLOCK
from .nf4 import dequantize_nf4
from .frozen_head import head_active_tiles


class _SparseFrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj):
        gate = gate_proj(x.to(torch.float16))
        up = up_proj(x.to(torch.float16))
        out = down_proj(swiglu(gate, up))
        ctx.save_for_backward(gate, up)
        ctx.layers = (gate_proj, up_proj, down_proj)
        ctx.input_shape = x.shape
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        gate_proj, up_proj, down_proj = ctx.layers
        g = grad.to(torch.float16).reshape(-1, ctx.input_shape[-1]).contiguous()
        rows, width = g.shape
        parts = triton.cdiv(width, 4096)
        flags = torch.empty((rows, parts), device=g.device, dtype=torch.int32)
        head_active_tiles[(rows, parts)](g, flags, width, parts, g.stride(0), 4096, num_warps=4)
        indices = flags.any(dim=1).nonzero(as_tuple=True)[0]
        dx = torch.zeros((rows, width), dtype=grad.dtype, device=grad.device)
        if not indices.numel():
            return dx.reshape(ctx.input_shape), None, None, None
        all_rows = indices.numel() == rows
        selected = g if all_rows else g.index_select(0, indices)
        gate = gate.reshape(rows, -1)
        up = up.reshape(rows, -1)
        if not all_rows:
            gate = gate.index_select(0, indices)
            up = up.index_select(0, indices)
        wd = dequantize_nf4(down_proj.weight, down_proj.weight.quant_state)
        middle_grad = torch.matmul(selected, wd)
        del wd
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            middle_grad, gate, up, dg, du, n, BLOCK=BLOCK)
        wg = dequantize_nf4(gate_proj.weight, gate_proj.weight.quant_state)
        values = torch.matmul(dg, wg).float()
        del wg
        wu = dequantize_nf4(up_proj.weight, up_proj.weight.quant_state)
        values.add_(torch.matmul(du, wu).float())
        if all_rows:
            dx = values
        else:
            dx.index_copy_(0, indices, values)
        return dx.reshape(ctx.input_shape), None, None, None


def sparse_mlp(x, gate, up, down):
    return _SparseFrozenMLP.apply(x, gate, up, down)


def patch_sparse_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from bitsandbytes.nn import Linear4bit
    original = Qwen2MLP.forward
    if getattr(original, '_team_sparse_mlp', False):
        return

    def forward(self, x):
        layers = (self.gate_proj, self.up_proj, self.down_proj)
        if (x.is_cuda and x.dtype == torch.float32 and x.numel() > 0
                and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.float16
                and getattr(self.config, 'hidden_act', 'silu') == 'silu'
                and all(type(m) is Linear4bit and not m.weight.requires_grad
                        and m.bias is None and m.compute_dtype == torch.float16
                        and m.weight.quant_state is not None
                        and m.weight.quant_state.quant_type == 'nf4'
                        and m.weight.quant_state.dtype == torch.float16
                        and m.weight.is_contiguous() for m in layers)):
            return sparse_mlp(x, *layers)
        return original(self, x)

    forward._team_sparse_mlp = True
    Qwen2MLP.forward = forward

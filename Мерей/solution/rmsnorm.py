"""Original row-wise Qwen2 RMSNorm for the frozen k-bit norm weights.

This fuses the float32 casts, square/mean, scaling, and weight multiplication.
Backward retains the original input, frozen weight, and one inverse RMS per
row. The forward has no full-size float32 conversion or elementwise temporaries.

Semantics follow transformers v5.16.1 Qwen2RMSNorm, including the intermediate
cast *before* multiplying the norm weight. No third-party kernel code is used.
"""

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rmsnorm_forward_kernel(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr,
                                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    weight = tl.load(W + col, col < N, 0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    # Qwen casts the normalized value back to the input dtype before the
    # multiplication, even when k-bit preparation keeps W and Y in float32.
    normalized = (x * inverse_rms).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + col, normalized * weight, col < N)
    tl.store(R + row, inverse_rms)


@triton.jit
def soup_rmsnorm_backward_kernel(DY, X, W, R, DX, N: tl.constexpr,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, 0).to(tl.float32)
    weight = tl.load(W + col, mask, 0).to(tl.float32)
    dy = tl.load(DY + row * N + col, mask, 0).to(tl.float32)
    inverse_rms = tl.load(R + row)
    # MulBackward first returns the gradient of the cast activation in its
    # original dtype. CastBackward then widens it for the float32 RMS graph.
    g = (dy * weight).to(X.dtype.element_ty).to(tl.float32)
    projection = tl.sum(g * x, axis=0)
    variance_grad = projection * (-0.5 * inverse_rms * inverse_rms * inverse_rms)
    dx = g * inverse_rms + (2.0 * x) * (variance_grad / N)
    tl.store(DX + row * N + col, dx, mask)


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        width = x.shape[-1]
        rows = x.numel() // width
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
        inverse_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(width)
        warps = 4 if block <= 2048 else 8
        soup_rmsnorm_forward_kernel[(rows,)](
            x, weight, out, inverse_rms, width, float(eps), block,
            num_warps=warps, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, inverse_rms)
        ctx.width, ctx.block, ctx.warps = width, block, warps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, inverse_rms = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        soup_rmsnorm_backward_kernel[(x.numel() // ctx.width,)](
            grad_out.contiguous(), x, weight, inverse_rms, grad_x,
            ctx.width, ctx.block, num_warps=ctx.warps, enable_fp_fusion=False,
        )
        return grad_x, None, None


def _supported(x, weight):
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x.is_cuda and weight.is_cuda
        and x.device == weight.device
        and x.ndim >= 1 and x.numel() > 0
        and x.is_contiguous() and weight.is_contiguous()
        and weight.ndim == 1 and weight.numel() == x.shape[-1]
        and 0 < x.shape[-1] <= 8192
        and x.dtype in (torch.float16, torch.float32)
        and weight.dtype in (torch.float16, torch.float32)
        and not weight.requires_grad
    )


def frozen_rmsnorm(x, weight, eps=1e-6):
    """Match Qwen's RMSNorm; use eager autograd outside the supported path."""
    if not _supported(x, weight):
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps)
        return weight * normalized.to(x.dtype)
    return _FrozenRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patch only the real Qwen2 class; repeated apply() calls are harmless."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    original = Qwen2RMSNorm.forward
    if getattr(original, "_soup_original_rmsnorm", False):
        return

    @functools.wraps(original)
    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if not _supported(hidden_states, weight):
            return original(self, hidden_states)
        return _FrozenRMSNorm.apply(hidden_states, weight, self.variance_epsilon)

    forward._soup_original_rmsnorm = True
    Qwen2RMSNorm.forward = forward

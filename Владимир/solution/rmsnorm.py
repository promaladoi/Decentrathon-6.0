"""Fused RMSNorm for Qwen2, forward and backward in Triton.

Qwen2RMSNorm in PyTorch is six launches each way (cast, pow, mean, rsqrt, mul, mul), and every one
reads and writes the whole fp32 hidden state. Here one program normalizes one row (one token) in
registers: one read of x and one write of y forward, one read of x and dy and one write of dx
backward. Only x (which autograd keeps alive anyway as the residual) and one fp32 rstd per row are
saved for backward, instead of PyTorch's fp32 intermediates.

The math follows transformers 5.16.1 exactly, rounding included:
    x_hat = x.float() * rsqrt(mean(x.float()**2) + eps)
    y = weight * x_hat.to(x.dtype)          # dtype: promote(weight, x), as in PyTorch
Idea: the standard fused RMSNorm (as in Triton's layer-norm tutorial,
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html); code written by us.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(X, W, Y, RSTD, stride_x, stride_y, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(RSTD + row, rstd)
    x_hat = (x * rstd).to(X.dtype.element_ty)  # the .to(input_dtype) of the PyTorch version
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = w * x_hat.to(tl.float32)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(DY, X, W, RSTD, DX, stride_dy, stride_x, stride_dx, N,
                       BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    x_hat = x * rstd
    g = dy * w
    # d/dx of x * rstd(x): rstd * (g - x_hat * mean(g * x_hat))
    dx = rstd * (g - x_hat * (tl.sum(g * x_hat, axis=0) / N))
    tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)


def _launch_config(n):
    block = triton.next_power_of_2(n)
    return block, min(max(block // 256, 1), 16)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape, n = x.shape, x.shape[-1]
        x2 = x.reshape(-1, n)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        weight = weight.contiguous()
        y = torch.empty(x2.shape, device=x.device, dtype=torch.promote_types(weight.dtype, x.dtype))
        rstd = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
        block, warps = _launch_config(n)
        rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, y, rstd, x2.stride(0), y.stride(0), n, eps,
                                           BLOCK_N=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n = x2.shape[-1]
        dy2 = dy.reshape(-1, n)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty(x2.shape, device=x2.device, dtype=x2.dtype)
        block, warps = _launch_config(n)
        rmsnorm_bwd_kernel[(x2.shape[0],)](dy2, x2, weight, rstd, dx, dy2.stride(0), x2.stride(0),
                                           dx.stride(0), n, BLOCK_N=block, num_warps=warps)
        dw = None
        if ctx.needs_input_grad[1]:  # frozen under LoRA; kept so the function stays general
            x_hat = (x2.float() * rstd[:, None]).to(x2.dtype)
            dw = (dy2.float() * x_hat.float()).sum(0).to(weight.dtype)
        return dx.reshape(ctx.shape), dw, None


def rmsnorm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel (a class patch, like the example)."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda or hidden_states.shape[-1] > 16384:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward

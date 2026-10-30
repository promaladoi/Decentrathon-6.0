"""Fused RMSNorm for Qwen2, forward and backward in Triton, alone or with the residual add before it.

Qwen2RMSNorm in PyTorch is six launches each way (cast, pow, mean, rsqrt, mul, mul), and every one
reads and writes the whole fp32 hidden state. Here one program normalizes one row (one token) in
registers: one read of x and one write of y forward, one read of x and dy and one write of dx
backward. Only x (which autograd keeps alive anyway as the residual) and one fp32 rstd per row are
saved for backward, instead of PyTorch's fp32 intermediates.

add_rmsnorm also takes the residual add that precedes the post-attention norm in every decoder
layer (`h = residual + attn_out; y = norm(h)`): one launch reads both inputs and writes h and y.
Its backward adds the residual stream's own gradient to the norm's and writes it in both input
dtypes, which is what autograd does for the add.

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


@triton.jit
def add_rmsnorm_fwd_kernel(R, D, W, H, Y, RSTD, stride_r, stride_d, stride_h, stride_y, N, eps,
                           BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(D + row * stride_d + cols, mask=mask, other=0.0).to(tl.float32)
    h = (r + d).to(H.dtype.element_ty)  # residual + attn_out in their promoted dtype
    tl.store(H + row * stride_h + cols, h, mask=mask)
    x = h.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(RSTD + row, rstd)
    x_hat = (x * rstd).to(H.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = w * x_hat.to(tl.float32)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def add_rmsnorm_bwd_kernel(DY, DH, X, W, RSTD, DX, DD, stride_dy, stride_dh, stride_x, stride_dx,
                           stride_dd, N, BLOCK_N: tl.constexpr, HAS_DH: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    x_hat = x * rstd
    g = dy * w
    dx = rstd * (g - x_hat * (tl.sum(g * x_hat, axis=0) / N))
    if HAS_DH:  # h also feeds the next residual add: its gradient joins the norm's
        dx += tl.load(DH + row * stride_dh + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)
    # the same gradient for the attention output, in its own dtype (autograd's add backward)
    tl.store(DD + row * stride_dd + cols, dx.to(DD.dtype.element_ty), mask=mask)


def _launch_config(n):
    block = triton.next_power_of_2(n)
    return block, min(max(block // 256, 1), 16)


def _rows(t, n):
    t2 = t.reshape(-1, n)
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def supported(*tensors):
    """The dtypes and sizes these kernels are written for (bf16 has no T4 path here)."""
    return all(t.is_cuda and t.dtype in (torch.float32, torch.float16) for t in tensors) and \
        tensors[0].shape[-1] <= 16384


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


class AddRMSNorm(torch.autograd.Function):
    """(h, y) = (residual + delta, norm(residual + delta)) in one launch each way."""

    @staticmethod
    def forward(ctx, residual, delta, weight, eps):
        shape, n = residual.shape, residual.shape[-1]
        r2, d2 = _rows(residual, n), _rows(delta, n)
        weight = weight.contiguous()
        h_dtype = torch.promote_types(residual.dtype, delta.dtype)
        h = torch.empty(r2.shape, device=r2.device, dtype=h_dtype)
        y = torch.empty(r2.shape, device=r2.device, dtype=torch.promote_types(weight.dtype, h_dtype))
        rstd = torch.empty(r2.shape[0], device=r2.device, dtype=torch.float32)
        block, warps = _launch_config(n)
        add_rmsnorm_fwd_kernel[(r2.shape[0],)](r2, d2, weight, h, y, rstd, r2.stride(0), d2.stride(0),
                                               h.stride(0), y.stride(0), n, eps, BLOCK_N=block,
                                               num_warps=warps)
        ctx.save_for_backward(h, weight, rstd)
        ctx.shape, ctx.dtypes = shape, (residual.dtype, delta.dtype)
        return h.view(shape), y.view(shape)

    @staticmethod
    def backward(ctx, dh, dy):
        h, weight, rstd = ctx.saved_tensors
        n = h.shape[-1]
        r_dtype, d_dtype = ctx.dtypes
        if dy is None:  # y unused: the add's gradient only
            dh = torch.zeros(h.shape, device=h.device, dtype=h.dtype) if dh is None else dh
            return dh.to(r_dtype), dh.to(d_dtype), None, None
        dy2 = _rows(dy, n)
        dh2 = _rows(dh, n) if dh is not None else dy2
        dx = torch.empty(h.shape, device=h.device, dtype=r_dtype)
        dd = torch.empty(h.shape, device=h.device, dtype=d_dtype)
        block, warps = _launch_config(n)
        add_rmsnorm_bwd_kernel[(h.shape[0],)](dy2, dh2, h, weight, rstd, dx, dd, dy2.stride(0),
                                              dh2.stride(0), h.stride(0), dx.stride(0), dd.stride(0),
                                              n, BLOCK_N=block, HAS_DH=dh is not None,
                                              num_warps=warps)
        dw = None
        if ctx.needs_input_grad[2]:
            x_hat = (h.float() * rstd[:, None]).to(h.dtype)
            dw = (dy2.float() * x_hat.float()).sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dd.view(ctx.shape), dw, None


def rmsnorm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def add_rmsnorm(residual, delta, weight, eps):
    return AddRMSNorm.apply(residual, delta, weight, eps)


def make_rmsnorm_forward(original):
    def forward(self, hidden_states):
        if not supported(hidden_states):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    return forward


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel (a class patch, like the example)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    cls.forward = make_rmsnorm_forward(cls.forward)


def warmup(n=1536):
    """Compile the variants Qwen2.5-1.5B launches (fp32 hidden state, fp32 weights, fp16
    attention output) ahead of the first step; see warmup.py."""
    f32, f16 = torch.float32, torch.float16
    block, warps = _launch_config(n)
    cfg = dict(grid=(1,), BLOCK_N=block, num_warps=warps)
    rmsnorm_fwd_kernel.warmup(f32, f32, f32, f32, n, n, n, 1e-6, **cfg)
    add_rmsnorm_fwd_kernel.warmup(f32, f16, f32, f32, f32, f32, n, n, n, n, n, 1e-6, **cfg)
    rmsnorm_bwd_kernel.warmup(f32, f32, f32, f32, f32, n, n, n, n, **cfg)
    add_rmsnorm_bwd_kernel.warmup(f32, f32, f32, f32, f32, f32, f16, n, n, n, n, n, n,
                                  HAS_DH=True, **cfg)

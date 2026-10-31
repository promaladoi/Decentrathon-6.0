"""Rematerialize the LoRA-A input from a shared RMS residual and dropout bits.

Only the private, guarded packed training path calls this operation. Qualified
single GPU-RNG-thread calls reproduce native full-shape Philox coordinates;
all other consumers draw through native dropout. The saved FP16
LoRA input is replaced by references to the RMS producer's existing residual,
weight and inverse RMS. Reconstructing it adds no GEMM and draws no RNG.
Native FP16 matrix multiplications and all FP32 rounding boundaries are kept.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .dropout import (BLOCK, _scales, real_dropout_gather_kernel,
                      real_dropout_backward_kernel)

_NATIVE_LINEAR_FORWARD = torch.nn.Linear.forward


@triton.jit
def rms_dropout_rematerialize_kernel(H, W, Rstd, Mask, Out, N,
                                    WIDTH: tl.constexpr, SCALE: tl.constexpr,
                                    BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    h = tl.load(H + offset, valid, other=0.0).to(tl.float32)
    inverse = tl.load(Rstd + offset // WIDTH, valid, other=0.0).to(tl.float32)
    weight = tl.load(W + offset % WIDTH, valid, other=0.0).to(tl.float32)
    normalized = h * inverse
    value = weight * normalized
    byte = tl.load(Mask + offset // 8, valid, other=0).to(tl.int32)
    kept = ((byte >> (offset % 8)) & 1).to(tl.float32)
    masked = value * kept
    tl.store(Out + offset, masked * SCALE, valid)


def supported(x, linear, descriptor):
    """Narrow native AMP/frozen-RMS contract, checked before any random draw."""
    if (descriptor is None or type(linear) is not torch.nn.Linear
            or linear.bias is not None or not linear.training
            or not torch.is_grad_enabled() or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.float16
            or torch.nn.Linear.forward is not _NATIVE_LINEAR_FORWARD
            or getattr(linear.forward, '__func__', None) is not _NATIVE_LINEAR_FORWARD
            or _NATIVE_LINEAR_FORWARD.__module__ != 'torch.nn.modules.linear'
            or _NATIVE_LINEAR_FORWARD.__qualname__ != 'Linear.forward'
            or hasattr(_NATIVE_LINEAR_FORWARD, '__wrapped__')
            or hasattr(linear, '_hf_hook')
            or any(getattr(linear, name, None) for name in (
                '_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))):
        return False
    h, weight, rstd = descriptor
    a = linear.weight
    return (x.is_cuda and x.ndim == 2 and x.is_contiguous() and x.numel() > 0
            and x.dtype == torch.float32 and h.shape == x.shape
            and h.is_contiguous() and h.dtype == torch.float32
            and weight.shape == (x.shape[1],) and weight.is_contiguous()
            and weight.dtype == torch.float32 and not weight.requires_grad
            and rstd.shape == (x.shape[0],) and rstd.is_contiguous()
            and rstd.dtype == torch.float32 and not rstd.requires_grad
            and a.ndim == 2 and a.shape[1] == x.shape[1]
            and a.dtype == torch.float32 and a.is_contiguous()
            and a.requires_grad and all(t.device == x.device for t in (h,weight,rstd,a)))


class _LoRAInputProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_weight, descriptor, rows, batch, sequence, p):
        forward_scale, backward_scale = _scales(p)
        n = x.numel()
        from .direct_dropout import try_gather
        direct = try_gather(x,rows,batch,sequence,p)
        if direct is None:
            dummy = torch.empty((batch, sequence, x.shape[1]), device=x.device,
                                dtype=torch.float32, memory_format=torch.contiguous_format)
            if dummy.data_ptr() % 16:
                raise RuntimeError('native dropout requires aligned CUDA allocation')
            discarded, full_mask = torch.ops.aten.native_dropout.default(dummy, float(p), True)
            del discarded, dummy
            dropped_half = torch.empty(x.shape, device=x.device, dtype=torch.float16)
            mask = torch.empty((triton.cdiv(n,8),), device=x.device, dtype=torch.uint8)
            real_dropout_gather_kernel[(triton.cdiv(n,BLOCK),)](
                x, full_mask, rows, dropped_half, mask, n, batch*sequence,
                x.shape[1], forward_scale, BLOCK, num_warps=4, enable_fp_fusion=False)
        else:
            dropped_half,mask = direct
        a_half = a_weight.to(torch.float16)
        output = F.linear(dropped_half, a_half)
        h, weight, rstd = descriptor
        ctx.save_for_backward(a_half, h, weight, rstd, mask)
        ctx.forward_scale, ctx.backward_scale = forward_scale, backward_scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        a, h, weight, rstd, mask = ctx.saved_tensors
        grad = grad_output.to(torch.float16)
        dx = da = None
        n = h.numel()
        # Explicit dtypes keep the native autocast Linear's two half GEMMs.
        with torch.autocast('cuda', enabled=False):
            if ctx.needs_input_grad[1]:
                dropped_half = torch.empty(h.shape, device=h.device, dtype=torch.float16)
                rms_dropout_rematerialize_kernel[(triton.cdiv(n,BLOCK),)](
                    h, weight, rstd, mask, dropped_half, n, h.shape[1],
                    ctx.forward_scale, BLOCK, num_warps=4, enable_fp_fusion=False)
                da = torch.matmul(grad.t(), dropped_half).to(torch.float32)
                del dropped_half
            if ctx.needs_input_grad[0]:
                half_dx = torch.matmul(grad, a).contiguous()
                dx = torch.empty(h.shape, device=h.device, dtype=torch.float32)
                real_dropout_backward_kernel[(triton.cdiv(n,BLOCK),)](
                    half_dx, mask, dx, n, ctx.backward_scale, BLOCK,
                    num_warps=4, enable_fp_fusion=False)
        return dx, da, None, None, None, None, None


def lora_input_projection(x, a_weight, descriptor, rows, batch, sequence, p):
    """Internal caller has already qualified native dropout and RMS metadata."""
    return _LoRAInputProjection.apply(x,a_weight,descriptor,rows,batch,sequence,p)


def warmup(width=1536, p=0.05):
    """Compile arithmetic only, inside this measured command; no RNG draws."""
    from .warmup import load_compiled
    from .direct_dropout import warmup as warmup_direct
    warmup_direct(width,p)
    forward,backward = _scales(p)
    load_compiled(real_dropout_gather_kernel.warmup(
        torch.float32,torch.bool,torch.int64,torch.float16,torch.uint8,
        16*width,16,width,forward,BLOCK,grid=(1,),num_warps=4,enable_fp_fusion=False))
    load_compiled(rms_dropout_rematerialize_kernel.warmup(
        torch.float32,torch.float32,torch.float32,torch.uint8,torch.float16,
        16*width,width,forward,BLOCK,grid=(1,),num_warps=4,enable_fp_fusion=False))
    load_compiled(real_dropout_backward_kernel.warmup(
        torch.float16,torch.uint8,torch.float32,16*width,backward,BLOCK,
        grid=(1,),num_warps=4,enable_fp_fusion=False))

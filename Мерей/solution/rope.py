"""Original Qwen2 rotary-position kernels, including fp16 product rounding.

The fixed model calls RoPE on transposed projection views [B, H, S, D].
Read their strides directly rather than making dense Q/K copies. Backward
needs only the frozen cosine/sine tensors; it does not retain Q or K.
The formulas and cast order follow Transformers 5.16.1's Python operation.
"""

import functools

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def soup_rope_forward_kernel(Q, K, C, Sine, OQ, OK, SEQ,
                             HQ: tl.constexpr, D: tl.constexpr,
                             QB, QH, QS, KB, KH, KS,
                             CB, CS, SB, SS, OBQ, OHQ, OSQ, OBK, OHK, OSK,
                             BROADCAST: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = token // SEQ
    position = token % SEQ
    dim = tl.arange(0, BLOCK)
    half = D // 2
    mask = dim < half
    cbatch = 0 if BROADCAST else batch
    cbase = cbatch * CB + position * CS
    sbase = cbatch * SB + position * SS
    c0 = tl.load(C + cbase + dim, mask, 0).to(tl.float32)
    c1 = tl.load(C + cbase + dim + half, mask, 0).to(tl.float32)
    s0 = tl.load(Sine + sbase + dim, mask, 0).to(tl.float32)
    s1 = tl.load(Sine + sbase + dim + half, mask, 0).to(tl.float32)
    if head < HQ:
        offset = batch * QB + head * QH + position * QS
        q0 = tl.load(Q + offset + dim, mask, 0).to(tl.float32)
        q1 = tl.load(Q + offset + dim + half, mask, 0).to(tl.float32)
        # PyTorch stores each multiplication in its promoted dtype before
        # adding. In half precision a single final cast would differ.
        a0 = (q0 * c0).to(OQ.dtype.element_ty).to(tl.float32)
        a1 = (q1 * c1).to(OQ.dtype.element_ty).to(tl.float32)
        b0 = ((-q1) * s0).to(OQ.dtype.element_ty).to(tl.float32)
        b1 = (q0 * s1).to(OQ.dtype.element_ty).to(tl.float32)
        output = batch * OBQ + head * OHQ + position * OSQ
        tl.store(OQ + output + dim, a0 + b0, mask)
        tl.store(OQ + output + dim + half, a1 + b1, mask)
    else:
        khead = head - HQ
        offset = batch * KB + khead * KH + position * KS
        k0 = tl.load(K + offset + dim, mask, 0).to(tl.float32)
        k1 = tl.load(K + offset + dim + half, mask, 0).to(tl.float32)
        a0 = (k0 * c0).to(OK.dtype.element_ty).to(tl.float32)
        a1 = (k1 * c1).to(OK.dtype.element_ty).to(tl.float32)
        b0 = ((-k1) * s0).to(OK.dtype.element_ty).to(tl.float32)
        b1 = (k0 * s1).to(OK.dtype.element_ty).to(tl.float32)
        output = batch * OBK + khead * OHK + position * OSK
        tl.store(OK + output + dim, a0 + b0, mask)
        tl.store(OK + output + dim + half, a1 + b1, mask)


@triton.jit
def soup_rope_backward_kernel(GQ, GK, C, Sine, DQ, DK, SEQ,
                              HQ: tl.constexpr, D: tl.constexpr,
                              GQB, GQH, GQS, GQD, GKB, GKH, GKS, GKD,
                              CB, CS, SB, SS, DQB, DQH, DQS, DKB, DKH, DKS,
                              BROADCAST: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = token // SEQ
    position = token % SEQ
    dim = tl.arange(0, BLOCK)
    half = D // 2
    mask = dim < half
    cbatch = 0 if BROADCAST else batch
    cbase = cbatch * CB + position * CS
    sbase = cbatch * SB + position * SS
    c0 = tl.load(C + cbase + dim, mask, 0).to(tl.float32)
    c1 = tl.load(C + cbase + dim + half, mask, 0).to(tl.float32)
    s0 = tl.load(Sine + sbase + dim, mask, 0).to(tl.float32)
    s1 = tl.load(Sine + sbase + dim + half, mask, 0).to(tl.float32)
    if head < HQ:
        offset = batch * GQB + head * GQH + position * GQS
        g0 = tl.load(GQ + offset + dim * GQD, mask, 0).to(tl.float32)
        g1 = tl.load(GQ + offset + (dim + half) * GQD, mask, 0).to(tl.float32)
        # Each MulBackward casts its branch back to the input dtype before
        # autograd adds the direct and rotated paths.
        direct0 = (g0 * c0).to(DQ.dtype.element_ty).to(tl.float32)
        direct1 = (g1 * c1).to(DQ.dtype.element_ty).to(tl.float32)
        rotated0 = (g1 * s1).to(DQ.dtype.element_ty).to(tl.float32)
        rotated1 = (g0 * s0).to(DQ.dtype.element_ty).to(tl.float32)
        output = batch * DQB + head * DQH + position * DQS
        tl.store(DQ + output + dim, direct0 + rotated0, mask)
        tl.store(DQ + output + dim + half, direct1 - rotated1, mask)
    else:
        khead = head - HQ
        offset = batch * GKB + khead * GKH + position * GKS
        g0 = tl.load(GK + offset + dim * GKD, mask, 0).to(tl.float32)
        g1 = tl.load(GK + offset + (dim + half) * GKD, mask, 0).to(tl.float32)
        direct0 = (g0 * c0).to(DK.dtype.element_ty).to(tl.float32)
        direct1 = (g1 * c1).to(DK.dtype.element_ty).to(tl.float32)
        rotated0 = (g1 * s1).to(DK.dtype.element_ty).to(tl.float32)
        rotated1 = (g0 * s0).to(DK.dtype.element_ty).to(tl.float32)
        output = batch * DKB + khead * DKH + position * DKS
        tl.store(DK + output + dim, direct0 + rotated0, mask)
        tl.store(DK + output + dim + half, direct1 - rotated1, mask)


def _supported(q, k, cos, sin, unsqueeze_dim):
    tensors = (q, k, cos, sin)
    if unsqueeze_dim != 1 or not all(isinstance(x, torch.Tensor) for x in tensors):
        return False
    if not all(x.is_cuda and x.device == q.device for x in tensors):
        return False
    if q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3:
        return False
    if not all(x.dtype in (torch.float16, torch.float32) for x in tensors):
        return False
    if q.dtype != k.dtype or cos.dtype != sin.dtype or cos.shape != sin.shape:
        return False
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        return False
    batch, _heads, seq, width = q.shape
    if (min(q.shape) < 1 or min(k.shape) < 1 or width % 2
            or width > 512 or cos.shape[0] not in (1, batch)
            or cos.shape[1:] != (seq, width)):
        return False
    if cos.requires_grad or sin.requires_grad:
        return False
    # Projection transpose views are supported directly, as are other positive
    # stride inputs. Outputs have independent storage and their own strides.
    return all(x.stride(-1) == 1 and all(s > 0 for s in x.stride()) for x in tensors)


def _eager(q, k, cos, sin, unsqueeze_dim):
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    qhalf, khalf = q.shape[-1] // 2, k.shape[-1] // 2
    qrotate = torch.cat((-q[..., qhalf:], q[..., :qhalf]), dim=-1)
    krotate = torch.cat((-k[..., khalf:], k[..., :khalf]), dim=-1)
    return q * c + qrotate * s, k * c + krotate * s


class _RotaryPosition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        batch, heads, seq, width = q.shape
        output_dtype = torch.promote_types(q.dtype, cos.dtype)
        out_q = torch.empty_like(q, dtype=output_dtype)
        out_k = torch.empty_like(k, dtype=output_dtype)
        block = triton.next_power_of_2(width // 2)
        warps = 1 if block <= 128 else 4
        soup_rope_forward_kernel[(batch * seq, heads + k.shape[1])](
            q, k, cos, sin, out_q, out_k, seq, heads, width,
            *q.stride()[:3], *k.stride()[:3],
            *cos.stride()[:2], *sin.stride()[:2],
            *out_q.stride()[:3], *out_k.stride()[:3],
            cos.shape[0] == 1, block, num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(cos, sin)
        ctx.q_shape, ctx.k_shape = q.shape, k.shape
        ctx.q_dtype, ctx.k_dtype = q.dtype, k.dtype
        ctx.block, ctx.warps = block, warps
        ctx.set_materialize_grads(False)
        if not q.requires_grad:
            ctx.mark_non_differentiable(out_q)
        if not k.requires_grad:
            ctx.mark_non_differentiable(out_k)
        return out_q, out_k

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        need_q = ctx.needs_input_grad[0] and grad_q is not None
        need_k = ctx.needs_input_grad[1] and grad_k is not None
        if not (need_q or need_k):
            return None, None, None, None
        dq = torch.empty(ctx.q_shape, dtype=ctx.q_dtype, device=cos.device) if need_q else None
        dk = torch.empty(ctx.k_shape, dtype=ctx.k_dtype, device=cos.device) if need_k else None
        qheads = ctx.q_shape[1] if need_q else 0
        kheads = ctx.k_shape[1] if need_k else 0
        batch, _heads, seq, width = ctx.q_shape
        # Inactive branches have no grid programs; placeholders are never read.
        soup_rope_backward_kernel[(batch * seq, qheads + kheads)](
            grad_q if need_q else cos, grad_k if need_k else cos, cos, sin,
            dq if need_q else cos, dk if need_k else cos, seq, qheads, width,
            *(grad_q.stride() if need_q else (0, 0, 0, 0)),
            *(grad_k.stride() if need_k else (0, 0, 0, 0)),
            *cos.stride()[:2], *sin.stride()[:2],
            *(dq.stride()[:3] if need_q else (0, 0, 0)),
            *(dk.stride()[:3] if need_k else (0, 0, 0)),
            cos.shape[0] == 1, ctx.block, num_warps=ctx.warps,
            enable_fp_fusion=False)
        return dq, dk, None, None


def rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Qwen2 RoPE, with eager fallback outside the frozen CUDA contract."""
    if not _supported(q, k, cos, sin, unsqueeze_dim):
        return _eager(q, k, cos, sin, unsqueeze_dim)
    return _RotaryPosition.apply(q, k, cos, sin)


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_rope_patch", False):
        return

    @functools.wraps(original)
    def apply_rotary(q, k, cos, sin, unsqueeze_dim=1):
        if not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim)
        return _RotaryPosition.apply(q, k, cos, sin)

    apply_rotary._soup_rope_patch = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary

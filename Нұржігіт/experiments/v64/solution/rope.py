"""Rotary position embedding (RoPE) of Qwen2's query and key in Triton, forward and backward.

transformers rotates q and k with separate launches (two half slices, a negation, a concatenation,
two multiplies and an add for each of q and k) and every one is a full pass over a tensor; backward
repeats that through autograd. Here one program takes one token: it reads that position's cos and
sin once, rotates all 12 query heads and both key heads in registers, and writes each result once.
Backward is the same rotation transposed, with the same single pass.

The rounding is PyTorch's: each product is rounded to the output dtype before the add, and FMA
contraction is turned off, so the result equals transformers' apply_rotary_pos_emb
(q * cos + rotate_half(q) * sin). The output keeps the memory layout of the input, as PyTorch's
elementwise ops do. Only cos and sin, shared by all layers, are kept for backward.
Idea: RoPE (Su et al., RoFormer, https://arxiv.org/abs/2104.09864); code written by us.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_heads(X, OUT, sx_b, sx_h, sx_l, so_b, so_h, so_l, b, l, c1, c2, s1, s2,
                  H: tl.constexpr, HALF: tl.constexpr, BLOCK_H: tl.constexpr,
                  BACKWARD: tl.constexpr):
    heads = tl.arange(0, BLOCK_H)[:, None]
    cols = tl.arange(0, HALF)[None, :]
    mask = heads < H
    src = X + b * sx_b + l * sx_l + heads * sx_h + cols
    x1 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    dt = OUT.dtype.element_ty
    if BACKWARD:  # transpose of the rotation: dx1 = g1*c1 + g2*s2, dx2 = g2*c2 - g1*s1
        o1 = (x1 * c1).to(dt).to(tl.float32) + (x2 * s2).to(dt).to(tl.float32)
        o2 = (x2 * c2).to(dt).to(tl.float32) - (x1 * s1).to(dt).to(tl.float32)
    else:  # q*cos + rotate_half(q)*sin with rotate_half(q) = (-q2, q1)
        o1 = (x1 * c1).to(dt).to(tl.float32) - (x2 * s1).to(dt).to(tl.float32)
        o2 = (x2 * c2).to(dt).to(tl.float32) + (x1 * s2).to(dt).to(tl.float32)
    dst = OUT + b * so_b + l * so_l + heads * so_h + cols
    tl.store(dst, o1.to(dt), mask=mask)
    tl.store(dst + HALF, o2.to(dt), mask=mask)


@triton.jit(do_not_specialize=["seq_len"])
def rope_kernel(Q, K, COS, SIN, OQ, OK, sq_b, sq_h, sq_l, sk_b, sk_h, sk_l, soq_b, soq_h, soq_l,
                sok_b, sok_h, sok_l, sc_b, sc_l, seq_len, HQ: tl.constexpr, HK: tl.constexpr,
                HALF: tl.constexpr, BLOCK_HQ: tl.constexpr, BLOCK_HK: tl.constexpr,
                BACKWARD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // seq_len
    l = pid % seq_len
    cols = tl.arange(0, HALF)[None, :]
    pos = b * sc_b + l * sc_l + cols
    c1 = tl.load(COS + pos).to(tl.float32)
    c2 = tl.load(COS + pos + HALF).to(tl.float32)
    s1 = tl.load(SIN + pos).to(tl.float32)
    s2 = tl.load(SIN + pos + HALF).to(tl.float32)
    _rotate_heads(Q, OQ, sq_b, sq_h, sq_l, soq_b, soq_h, soq_l, b, l, c1, c2, s1, s2,
                  HQ, HALF, BLOCK_HQ, BACKWARD)
    _rotate_heads(K, OK, sk_b, sk_h, sk_l, sok_b, sok_h, sok_l, b, l, c1, c2, s1, s2,
                  HK, HALF, BLOCK_HK, BACKWARD)


def _launch(q, k, cos, oq, ok, sin, backward):
    batch, hq, seq_len, dim = q.shape
    hk = k.shape[1]
    sc_b = cos.stride(0) if cos.shape[0] > 1 else 0
    rope_kernel[(batch * seq_len,)](
        q, k, cos, sin, oq, ok, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
        k.stride(2), oq.stride(0), oq.stride(1), oq.stride(2), ok.stride(0), ok.stride(1),
        ok.stride(2), sc_b, cos.stride(1), seq_len, HQ=hq, HK=hk, HALF=dim // 2,
        BLOCK_HQ=triton.next_power_of_2(hq), BLOCK_HK=triton.next_power_of_2(hk),
        BACKWARD=backward, num_warps=4, enable_fp_fusion=False)


def _last_dim_dense(t):
    return t if t.stride(-1) == 1 else t.contiguous()


class RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q, k = _last_dim_dense(q), _last_dim_dense(k)
        oq, ok = torch.empty_like(q), torch.empty_like(k)  # same memory layout as PyTorch's result
        _launch(q, k, cos, oq, ok, sin, backward=False)
        ctx.save_for_backward(cos, sin)
        ctx.shapes = (q.shape, k.shape)
        return oq, ok

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        ref = gq if gq is not None else gk
        # an unused output has no gradient: rotate zeros for it
        gq = _last_dim_dense(gq) if gq is not None else ref.new_zeros(ctx.shapes[0])
        gk = _last_dim_dense(gk) if gk is not None else ref.new_zeros(ctx.shapes[1])
        dq, dk = torch.empty_like(gq), torch.empty_like(gk)
        _launch(gq, gk, cos, dq, dk, sin, backward=True)
        return dq, dk, None, None


def _supported(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or q.dim() != 4 or k.dim() != 4 or cos.dim() != 3:
        return False
    if not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if q.dtype not in (torch.float32, torch.float16) or k.dtype != q.dtype:
        return False
    if cos.dtype != q.dtype or sin.dtype != q.dtype or sin.shape != cos.shape or \
            sin.stride() != cos.stride():
        return False
    batch, _, seq_len, dim = q.shape
    half = dim // 2
    return (k.shape[0] == batch and k.shape[2] == seq_len and k.shape[3] == dim
            and cos.shape[0] in (1, batch) and cos.shape[1] == seq_len and cos.shape[2] == dim
            and cos.stride(2) == 1 and dim % 2 == 0 and half & (half - 1) == 0)


def make_apply_rotary_pos_emb(original):
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim)
        return RoPE.apply(q, k, cos, sin)

    return apply_rotary_pos_emb


def patch_qwen2_rope():
    """Qwen2Attention.forward looks up apply_rotary_pos_emb in its module at call time."""
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.apply_rotary_pos_emb = make_apply_rotary_pos_emb(
        modeling_qwen2.apply_rotary_pos_emb)


def warmup(hq=12, hk=2, dim=128):
    """Compile the fp32 forward and backward variants Qwen2.5-1.5B launches; see warmup.py."""
    from .warmup import load_compiled

    f32 = torch.float32
    n = 16  # every stride of these layouts is a multiple of 16; seq_len is not specialized
    for backward in (False, True):
        load_compiled(rope_kernel.warmup(f32, f32, f32, f32, f32, f32, n, n, n, n, n, n, n, n, n, n, n, n, 0, n,
                           7, grid=(1,), HQ=hq, HK=hk, HALF=dim // 2,
                           BLOCK_HQ=triton.next_power_of_2(hq), BLOCK_HK=triton.next_power_of_2(hk),
                           BACKWARD=backward, num_warps=4, enable_fp_fusion=False))

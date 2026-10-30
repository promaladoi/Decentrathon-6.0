"""Training forward of Qwen2 + 4-bit LoRA on the real tokens only, with pad tokens removed.

TRL pads the 4 rows of a batch on the right to the longest row. The rows average about 220 tokens
and one row of 512 sets the length for all four, so a large share of every batch is padding, and
transformers runs every projection, norm, MLP and the lm_head over it. Here the real tokens (the
attention mask) are gathered into one (tokens, hidden) tensor right after the embedding, and every
row-wise operation runs on that. Removing pad tokens inside a batch is allowed by the task; rows
of different batches are never mixed.

The packed candidate keeps attention on real tokens too. A team Triton kernel adds LoRA,
applies RoPE and writes fp16 Q/K/V directly in [1, heads, real_tokens, head_dim], with explicit
GQA repetition. A view feeds PyTorch2.11's native efficient/Cutlass varlen operator, using CUDA
int32 cumulative row boundaries and the original per-row causal rule. Its contiguous real-token
output is a view for o_proj; no padded Q/K/V zero fill or output gather/scatter is needed.
The legacy RoPEPad and UnpadHeads helpers remain for correctness comparisons and old APIs.

The arithmetic of every real token follows transformers + PEFT + bitsandbytes step by step:
x is cast to fp16 for each 4-bit projection, bias added by the same matmul, LoRA computed by the
same modules under the same autocast and added in fp32, RoPE in fp32 with PyTorch's rounding.
LoRA dropout runs on the full (batch, seq_len, hidden) tensor, the shape it has in the base run,
so the random number generator produces the same dropout masks in the same order.
The model output goes through lmhead.py: no logits of the vocabulary size are materialized.

Anything this file does not handle (inference, a cache, other attention implementations, DoRA,
gradient checkpointing, hooks on the bypassed modules, ...) runs the original forward.
Original unpadding integration by teammate Vladimir, commit987da36e1efa90652e1a7f8c48afef50685dbc40.
Packed integration by the same team; native operator/source pin is PyTorch v2.11.0
aten/src/ATen/native/transformers/cuda/attention.cu and derivatives.yaml. GPU qualification pending.
"""

import torch
import triton
import triton.language as tl

from .lmhead import LazyLogits, lm_head_loss, patch_metric_readers
from .mlp import MLP4bit, _compute_dtype
from .nf4 import MatMulNF4, dequantize_nf4, supported as nf4_supported, _linear_forward
from . import warmup as _warmup
from .rmsnorm import add_rmsnorm
from .dropout import real_row_dropout, _supported as _real_dropout_supported

# Static candidate configuration: no harness/test/environment-dependent switch.
REAL_ROW_DROPOUT = True
REAL_ROW_DROPOUT_MIN_PADDED_ELEMENTS = 4 * 384 * 1536
_NATIVE_DROPOUT_FORWARD = torch.nn.Dropout.forward


@triton.jit(do_not_specialize=["seq_len"])
def rope_pad_fwd_kernel(Q, LQ, K, LK, V, LV, COS, SIN, ROWS, QP, KP, VP, sp_b, sp_h, sp_l,
                        sk_b, sk_h, sk_l, seq_len, HQ: tl.constexpr, HK: tl.constexpr,
                        HALF: tl.constexpr,
                        REP: tl.constexpr, BLOCK_HQ: tl.constexpr, BLOCK_HK: tl.constexpr,
                        HAS_LQ: tl.constexpr, HAS_LK: tl.constexpr, HAS_LV: tl.constexpr,
                        PACKED: tl.constexpr = False):
    t = tl.program_id(0).to(tl.int64)
    if PACKED:
        b = 0
        l = t
    else:
        r = tl.load(ROWS + t)  # b * seq_len + l in the padded batch
        b = r // seq_len
        l = r % seq_len
    D = HALF * 2
    cols = tl.arange(0, HALF)[None, :]
    c1 = tl.load(COS + t * D + cols).to(tl.float32)
    c2 = tl.load(COS + t * D + HALF + cols).to(tl.float32)
    s1 = tl.load(SIN + t * D + cols).to(tl.float32)
    s2 = tl.load(SIN + t * D + HALF + cols).to(tl.float32)
    dt = QP.dtype.element_ty

    # q: projection + LoRA in fp32, rotated, cast, into (b, h, l)
    hq = tl.arange(0, BLOCK_HQ)[:, None]
    mq = hq < HQ
    src = t * (HQ * D) + hq * D + cols
    x1 = tl.load(Q + src, mask=mq, other=0.0).to(tl.float32)
    x2 = tl.load(Q + src + HALF, mask=mq, other=0.0).to(tl.float32)
    if HAS_LQ:
        x1 = x1 + tl.load(LQ + src, mask=mq, other=0.0).to(tl.float32)
        x2 = x2 + tl.load(LQ + src + HALF, mask=mq, other=0.0).to(tl.float32)
    dst = b * sp_b + hq * sp_h + l * sp_l + cols
    tl.store(QP + dst, ((x1 * c1) - (x2 * s1)).to(dt), mask=mq)
    tl.store(QP + dst + HALF, ((x2 * c2) + (x1 * s2)).to(dt), mask=mq)

    # k: the same rotation, written to the REP query heads that share it
    hk = tl.arange(0, BLOCK_HK)[:, None]
    mk = hk < HK
    srck = t * (HK * D) + hk * D + cols
    k1 = tl.load(K + srck, mask=mk, other=0.0).to(tl.float32)
    k2 = tl.load(K + srck + HALF, mask=mk, other=0.0).to(tl.float32)
    if HAS_LK:
        k1 = k1 + tl.load(LK + srck, mask=mk, other=0.0).to(tl.float32)
        k2 = k2 + tl.load(LK + srck + HALF, mask=mk, other=0.0).to(tl.float32)
    ko1 = ((k1 * c1) - (k2 * s1)).to(dt)
    ko2 = ((k2 * c2) + (k1 * s2)).to(dt)
    v1 = tl.load(V + srck, mask=mk, other=0.0).to(tl.float32)
    v2 = tl.load(V + srck + HALF, mask=mk, other=0.0).to(tl.float32)
    if HAS_LV:
        v1 = v1 + tl.load(LV + srck, mask=mk, other=0.0).to(tl.float32)
        v2 = v2 + tl.load(LV + srck + HALF, mask=mk, other=0.0).to(tl.float32)
    vo1 = v1.to(dt)
    vo2 = v2.to(dt)
    for rep in tl.static_range(REP):
        dstk = b * sk_b + (hk * REP + rep) * sk_h + l * sk_l + cols  # KP, VP: same layout
        tl.store(KP + dstk, ko1, mask=mk)
        tl.store(KP + dstk + HALF, ko2, mask=mk)
        tl.store(VP + dstk, vo1, mask=mk)
        tl.store(VP + dstk + HALF, vo2, mask=mk)


@triton.jit(do_not_specialize=["seq_len"])
def rope_pad_bwd_kernel(DQP, DKP, DVP, COS, SIN, ROWS, DQ, DLQ, DK, DLK, DV, DLV, sq_b, sq_h,
                        sq_l, sk_b, sk_h, sk_l, sv_b, sv_h, sv_l, seq_len, HQ: tl.constexpr,
                        HK: tl.constexpr, HALF: tl.constexpr, REP: tl.constexpr,
                        BLOCK_HQ: tl.constexpr, BLOCK_HK: tl.constexpr, HAS_LQ: tl.constexpr,
                        HAS_LK: tl.constexpr, HAS_LV: tl.constexpr,
                        PACKED: tl.constexpr = False):
    t = tl.program_id(0).to(tl.int64)
    if PACKED:
        b = 0
        l = t
    else:
        r = tl.load(ROWS + t)
        b = r // seq_len
        l = r % seq_len
    D = HALF * 2
    cols = tl.arange(0, HALF)[None, :]
    c1 = tl.load(COS + t * D + cols).to(tl.float32)
    c2 = tl.load(COS + t * D + HALF + cols).to(tl.float32)
    s1 = tl.load(SIN + t * D + cols).to(tl.float32)
    s2 = tl.load(SIN + t * D + HALF + cols).to(tl.float32)
    dt = DQ.dtype.element_ty

    # q: the transposed rotation of the attention gradient; the same gradient reaches the
    # projection and its LoRA output (autograd's add backward)
    hq = tl.arange(0, BLOCK_HQ)[:, None]
    mq = hq < HQ
    src = b * sq_b + hq * sq_h + l * sq_l + cols
    g1 = tl.load(DQP + src, mask=mq, other=0.0).to(tl.float32)
    g2 = tl.load(DQP + src + HALF, mask=mq, other=0.0).to(tl.float32)
    d1 = ((g1 * c1) + (g2 * s2)).to(dt)
    d2 = ((g2 * c2) - (g1 * s1)).to(dt)
    dst = t * (HQ * D) + hq * D + cols
    tl.store(DQ + dst, d1, mask=mq)
    tl.store(DQ + dst + HALF, d2, mask=mq)
    if HAS_LQ:
        tl.store(DLQ + dst, d1, mask=mq)
        tl.store(DLQ + dst + HALF, d2, mask=mq)

    # k and v: the REP query heads' gradients summed in fp32 (repeat_kv's backward)
    hk = tl.arange(0, BLOCK_HK)[:, None]
    mk = hk < HK
    gk1 = tl.zeros([BLOCK_HK, HALF], dtype=tl.float32)
    gk2 = tl.zeros([BLOCK_HK, HALF], dtype=tl.float32)
    gv1 = tl.zeros([BLOCK_HK, HALF], dtype=tl.float32)
    gv2 = tl.zeros([BLOCK_HK, HALF], dtype=tl.float32)
    for rep in tl.static_range(REP):
        sk = b * sk_b + (hk * REP + rep) * sk_h + l * sk_l + cols
        sv = b * sv_b + (hk * REP + rep) * sv_h + l * sv_l + cols
        gk1 += tl.load(DKP + sk, mask=mk, other=0.0).to(tl.float32)
        gk2 += tl.load(DKP + sk + HALF, mask=mk, other=0.0).to(tl.float32)
        gv1 += tl.load(DVP + sv, mask=mk, other=0.0).to(tl.float32)
        gv2 += tl.load(DVP + sv + HALF, mask=mk, other=0.0).to(tl.float32)
    dk1 = ((gk1 * c1) + (gk2 * s2)).to(dt)
    dk2 = ((gk2 * c2) - (gk1 * s1)).to(dt)
    dstk = t * (HK * D) + hk * D + cols
    tl.store(DK + dstk, dk1, mask=mk)
    tl.store(DK + dstk + HALF, dk2, mask=mk)
    if HAS_LK:
        tl.store(DLK + dstk, dk1, mask=mk)
        tl.store(DLK + dstk + HALF, dk2, mask=mk)
    tl.store(DV + dstk, gv1.to(dt), mask=mk)
    tl.store(DV + dstk + HALF, gv2.to(dt), mask=mk)
    if HAS_LV:
        tl.store(DLV + dstk, gv1.to(dt), mask=mk)
        tl.store(DLV + dstk + HALF, gv2.to(dt), mask=mk)


@triton.jit(do_not_specialize=["seq_len", "to_flat"])  # one compiled variant, both directions
def heads_rows_kernel(FLAT, PAD, ROWS, sp_b, sp_h, sp_l, seq_len, to_flat, H: tl.constexpr,
                      D: tl.constexpr, BLOCK_H: tl.constexpr):
    """(batch, heads, seq_len, D) <-> (tokens, heads * D) for the real tokens."""
    t = tl.program_id(0).to(tl.int64)
    r = tl.load(ROWS + t)
    b = r // seq_len
    l = r % seq_len
    heads = tl.arange(0, BLOCK_H)[:, None]
    cols = tl.arange(0, D)[None, :]
    mask = heads < H
    flat = FLAT + t * (H * D) + heads * D + cols
    pad = PAD + b * sp_b + heads * sp_h + l * sp_l + cols
    if to_flat:
        tl.store(flat, tl.load(pad, mask=mask, other=0.0), mask=mask)
    else:
        tl.store(pad, tl.load(flat, mask=mask, other=0.0), mask=mask)


def _blocks(hq, hk):
    return triton.next_power_of_2(hq), triton.next_power_of_2(hk)


class RoPEPad(torch.autograd.Function):
    """(q, k, v) of the real tokens (+ LoRA outputs) -> rotated, fp16, padded, GQA-repeated."""

    @staticmethod
    def forward(ctx, q, lq, k, lk, v, lv, cos, sin, rows, batch, seq_len, hq, hk, dim, rep):
        dtype = torch.float16
        out_heads = hk * rep
        packed = bool(getattr(ctx, "_packed", False))
        if packed:
            if out_heads != hq:
                raise ValueError("packed efficient attention requires explicit equal heads")
            layout_batch, layout_sequence = 1, rows.numel()
            allocate = q.new_empty  # every real element is written exactly once
        else:
            layout_batch, layout_sequence = batch, seq_len
            allocate = q.new_zeros
        qp = allocate((layout_batch, hq, layout_sequence, dim), dtype=dtype)
        kp = allocate((layout_batch, out_heads, layout_sequence, dim), dtype=dtype)
        vp = allocate((layout_batch, out_heads, layout_sequence, dim), dtype=dtype)
        bq, bk = _blocks(hq, hk)
        rope_pad_fwd_kernel[(rows.numel(),)](
            q, lq if lq is not None else q, k, lk if lk is not None else k, v,
            lv if lv is not None else v, cos, sin, rows, qp, kp, vp, qp.stride(0), qp.stride(1),
            qp.stride(2), kp.stride(0), kp.stride(1), kp.stride(2), seq_len, HQ=hq, HK=hk,
            HALF=dim // 2, REP=rep, BLOCK_HQ=bq,
            BLOCK_HK=bk, HAS_LQ=lq is not None, HAS_LK=lk is not None, HAS_LV=lv is not None,
            PACKED=packed, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(cos, sin, rows)
        ctx.meta = (layout_batch, layout_sequence, hq, hk, dim, rep, lq is not None, lk is not None,
                    lv is not None)
        ctx._packed = packed
        ctx.dtypes = (q.dtype, k.dtype, v.dtype)
        return qp, kp, vp

    @staticmethod
    def backward(ctx, dqp, dkp, dvp):
        cos, sin, rows = ctx.saved_tensors
        batch, seq_len, hq, hk, dim, rep, has_lq, has_lk, has_lv = ctx.meta
        tokens = rows.numel()
        f16 = torch.float16  # an output attention did not use gets a zero gradient
        dqp = dqp if dqp is not None else cos.new_zeros((batch, hq, seq_len, dim), dtype=f16)
        dkp = dkp if dkp is not None else cos.new_zeros((batch, hk * rep, seq_len, dim), dtype=f16)
        dvp = dvp if dvp is not None else cos.new_zeros((batch, hk * rep, seq_len, dim), dtype=f16)
        dqp, dkp, dvp = (t if t.stride(-1) == 1 else t.contiguous() for t in (dqp, dkp, dvp))
        dt = torch.float16
        dq = torch.empty((tokens, hq * dim), device=cos.device, dtype=dt)
        dk = torch.empty((tokens, hk * dim), device=cos.device, dtype=dt)
        dv = torch.empty((tokens, hk * dim), device=cos.device, dtype=dt)
        dlq = torch.empty_like(dq) if has_lq else None
        dlk = torch.empty_like(dk) if has_lk else None
        dlv = torch.empty_like(dv) if has_lv else None
        bq, bk = _blocks(hq, hk)
        rope_pad_bwd_kernel[(tokens,)](
            dqp, dkp, dvp, cos, sin, rows, dq, dlq if has_lq else dq, dk, dlk if has_lk else dk,
            dv, dlv if has_lv else dv, dqp.stride(0), dqp.stride(1), dqp.stride(2),
            dkp.stride(0), dkp.stride(1), dkp.stride(2), dvp.stride(0), dvp.stride(1),
            dvp.stride(2), seq_len, HQ=hq, HK=hk, HALF=dim // 2, REP=rep, BLOCK_HQ=bq,
            BLOCK_HK=bk, HAS_LQ=has_lq, HAS_LK=has_lk, HAS_LV=has_lv, num_warps=4,
            PACKED=bool(getattr(ctx, "_packed", False)), enable_fp_fusion=False)
        q_dtype, k_dtype, v_dtype = ctx.dtypes
        return (dq.to(q_dtype), dlq, dk.to(k_dtype), dlk, dv.to(v_dtype), dlv,
                None, None, None, None, None, None, None, None, None)


class RoPEPacked(torch.autograd.Function):
    """Same input/gradient API as RoPEPad, returning [1,H,real_tokens,D]."""

    @staticmethod
    def forward(ctx, q, lq, k, lk, v, lv, cos, sin, rows, batch, seq_len, hq, hk, dim, rep):
        ctx._packed = True
        return RoPEPad.forward(ctx, q, lq, k, lk, v, lv, cos, sin, rows,
                               batch, seq_len, hq, hk, dim, rep)

    @staticmethod
    def backward(ctx, dqp, dkp, dvp):
        return RoPEPad.backward(ctx, dqp, dkp, dvp)


class UnpadHeads(torch.autograd.Function):
    """(batch, heads, seq_len, D) attention output -> (tokens, heads * D) of the real tokens."""

    @staticmethod
    def forward(ctx, padded, rows, seq_len):
        padded = padded if padded.stride(-1) == 1 else padded.contiguous()
        batch, heads, _, dim = padded.shape
        flat = padded.new_empty((rows.numel(), heads * dim))
        _heads_rows(flat, padded, rows, seq_len, to_flat=True)
        ctx.save_for_backward(rows)
        ctx.meta = (batch, heads, seq_len, dim)
        return flat

    @staticmethod
    def backward(ctx, grad_flat):
        (rows,) = ctx.saved_tensors
        batch, heads, seq_len, dim = ctx.meta
        grad_flat = grad_flat.contiguous()
        padded = grad_flat.new_zeros((batch, heads, seq_len, dim))
        _heads_rows(grad_flat, padded, rows, seq_len, to_flat=False)
        return padded, None, None


def _heads_rows(flat, padded, rows, seq_len, to_flat):
    heads, dim = padded.shape[1], padded.shape[3]
    heads_rows_kernel[(rows.numel(),)](flat, padded, rows, padded.stride(0), padded.stride(1),
                                       padded.stride(2), seq_len, 1 if to_flat else 0, H=heads,
                                       D=dim, BLOCK_H=triton.next_power_of_2(heads), num_warps=4)


class QKV4bit(torch.autograd.Function):
    """The q, k, v base projections of one input: x cast to fp16 once, three NF4 matmuls with
    bias. Backward sums the three input gradients in fp32, as autograd sums the gradients of
    the three casts Linear4bit makes."""

    @staticmethod
    def forward(ctx, x, q_layer, k_layer, v_layer, dtype):
        x16 = x.to(dtype)
        outs = []
        for layer in (q_layer, k_layer, v_layer):
            bias = layer.bias.to(dtype) if layer.bias is not None else None
            # Preserve the pinned native small-M dispatch, and use the exact
            # one-pass decode for its ordinary dequantize+linear fallback.
            outs.append(_linear_forward(x16, layer.weight, bias, layer.weight.quant_state))
        ctx.layers, ctx.dtype, ctx.x_dtype = (q_layer, k_layer, v_layer), dtype, x.dtype
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        dx = None
        for layer, grad in zip(ctx.layers, grads):
            if grad is None:
                continue
            part = torch.matmul(grad.to(ctx.dtype), _weight(layer, ctx.dtype)).float()
            dx = part if dx is None else dx.add_(part)
        return dx.to(ctx.x_dtype) if dx is not None else None, None, None, None, None


def _weight(layer, dtype):
    return dequantize_nf4(layer.weight, layer.weight.quant_state, dtype)


# ---------------------------------------------------------------- model inspection

def _lora_parts(module):
    """(base Linear4bit, lora_A, lora_B, dropout, scaling) of a vanilla PEFT LoRA 4-bit layer
    with one active adapter; (module, None...) for a plain Linear4bit; None otherwise."""
    import bitsandbytes as bnb

    if isinstance(module, bnb.nn.Linear4bit):
        return module, None, None, None, None
    try:
        from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
    except ImportError:
        return None
    if not isinstance(module, LoraLinear4bit) or module.merged or module.disable_adapters:
        return None
    active = [a for a in module.active_adapters if a in module.lora_A.keys()]
    if len(active) != 1 or active[0] in module.lora_variant:
        return None
    name = active[0]
    lora_a, lora_b = module.lora_A[name], module.lora_B[name]
    dropout = module.lora_dropout[name]
    if type(lora_a) is not torch.nn.Linear or type(lora_b) is not torch.nn.Linear or \
            lora_a.bias is not None or lora_b.bias is not None:
        return None
    if type(dropout) not in (torch.nn.Dropout, torch.nn.Identity):
        return None
    base = module.base_layer
    if not isinstance(base, bnb.nn.Linear4bit):
        return None
    return base, lora_a, lora_b, dropout, module.scaling[name]


def _no_hooks(*modules):
    """No torch hooks and no accelerate hook (which replaces forward) on modules the fast
    forward calls through their weights instead of their forward."""
    module_impl = torch.nn.modules.module
    if any(getattr(module_impl, name, None) for name in
           ("_global_forward_hooks", "_global_forward_pre_hooks",
            "_global_backward_hooks", "_global_backward_pre_hooks")):
        return False
    return all(not (m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or
                    getattr(m, "_backward_pre_hooks", None) or hasattr(m, "_hf_hook"))
               for m in modules)


def _inspect(model):
    """Everything the fast forward needs, or None when the model is not the case it handles."""
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen2 import modeling_qwen2

    cfg = model.config
    inner = model.model
    n_layers = cfg.num_hidden_layers
    if (not torch.__version__.split("+")[0].startswith("2.11.")
            or getattr(cfg, "_attn_implementation", None) != "sdpa"
            or not torch.backends.cuda.mem_efficient_sdp_enabled()
            or inner.gradient_checkpointing
            or float(getattr(cfg, "attention_dropout", 0.0)) != 0.0):
        return None
    if any(t != "full_attention" for t in getattr(cfg, "layer_types", ["full_attention"] * n_layers)):
        return None
    if getattr(inner.rotary_emb, "rope_type", "default") != "default":
        return None
    if model.loss_function is not ForCausalLMLoss:
        return None
    head = model.lm_head
    if (type(head) is not torch.nn.Linear or head.bias is not None
            or not head.weight.is_cuda or head.weight.requires_grad
            or type(inner.norm) is not modeling_qwen2.Qwen2RMSNorm
            or inner.norm.weight.requires_grad or not inner.norm.weight.is_cuda
            or inner.embed_tokens.weight.requires_grad):
        return None
    hq, hk = cfg.num_attention_heads, cfg.num_key_value_heads
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // hq
    if hq % hk or dim % 8 or (dim // 2) & (dim // 2 - 1) or dim > 256:
        return None
    if not _no_hooks(inner.norm, head, inner.rotary_emb):
        return None
    layers = []
    frozen_modules = [head, inner.norm, inner.embed_tokens]
    watched_modules = [inner.norm, head, inner.rotary_emb, inner.embed_tokens]
    attention_modules = []
    for layer in inner.layers[:n_layers]:
        if type(layer) is not modeling_qwen2.Qwen2DecoderLayer:
            return None
        attn, mlp = layer.self_attn, layer.mlp
        if float(getattr(attn, "attention_dropout", getattr(cfg, "attention_dropout", 0.0))) != 0.0:
            return None
        if not getattr(attn, "is_causal", True):
            return None
        norms = (layer.input_layernorm, layer.post_attention_layernorm)
        if any(type(n) is not modeling_qwen2.Qwen2RMSNorm or n.weight.requires_grad
               or not n.weight.is_cuda for n in norms):
            return None
        if type(mlp) is not modeling_qwen2.Qwen2MLP or \
                getattr(cfg, "hidden_act", "silu") != "silu":
            return None
        projection_modules = [getattr(attn, name)
                              for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        projections = [_lora_parts(module) for module in projection_modules]
        if any(p is None for p in projections):
            return None
        bases = [p[0] for p in projections]
        dtype = _compute_dtype((mlp.gate_proj, mlp.up_proj, mlp.down_proj))
        if dtype != torch.float16:
            return None
        for base in bases:
            state = getattr(base.weight, "quant_state", None)
            if base.compute_dtype != torch.float16 or not nf4_supported(state) or \
                    not base.weight.is_cuda or any(p.requires_grad for p in base.parameters()):
                return None
        if bases[3].bias is not None:
            return None
        lora_modules = [module for parts in projections for module in parts[1:4]
                        if module is not None]
        if not _no_hooks(layer, attn, mlp, *norms, *bases, *projection_modules,
                         *lora_modules, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            return None
        layers.append((layer, projections))
        frozen_modules.extend((*norms, *bases, mlp.gate_proj, mlp.up_proj, mlp.down_proj))
        watched_modules.extend((layer, attn, mlp, *norms, *bases,
                                *projection_modules, *lora_modules,
                                mlp.gate_proj, mlp.up_proj, mlp.down_proj))
        attention_modules.append(attn)
    return dict(layers=layers, hq=hq, hk=hk, dim=dim, rep=hq // hk,
                frozen_modules=tuple(dict.fromkeys(frozen_modules)),
                watched_modules=tuple(dict.fromkeys(watched_modules)),
                attention_modules=tuple(attention_modules))


def _layout(model):
    cached = model.__dict__.get("_solution_unpad")
    if cached is None:
        try:
            cached = _inspect(model) or False
        except Exception:
            cached = False
        model.__dict__["_solution_unpad"] = cached
    if cached:
        # The fixed task keeps these parameters frozen. Recheck cheap mutable
        # flags/hooks before reusing its structural layout cache.
        if (float(getattr(model.config, "attention_dropout", 0.0)) != 0.0
                or any(float(getattr(a, "attention_dropout", 0.0)) != 0.0
                       for a in cached["attention_modules"])
                or any(not getattr(a, "is_causal", True) for a in cached["attention_modules"])
                or not torch.backends.cuda.mem_efficient_sdp_enabled()
                or not _no_hooks(*cached["watched_modules"])
                or any(getattr(m, key, None) is not None
                       and getattr(m, key).requires_grad
                       for m in cached["frozen_modules"] for key in ("weight", "bias"))):
            return None
    return cached or None


@torch.no_grad()
def _batch_metadata(input_ids, attention_mask, labels):
    """Validate packed-prefix semantics once, before any embedding/dropout work."""
    if (not all(isinstance(t, torch.Tensor) and t.is_cuda for t in
                (input_ids, attention_mask, labels))
            or input_ids.ndim != 2 or input_ids.dtype != torch.int64
            or labels.dtype != torch.int64 or input_ids.shape != labels.shape
            or attention_mask.shape != input_ids.shape
            or not (input_ids.device == attention_mask.device == labels.device)
            or attention_mask.dtype not in (torch.bool, torch.int32, torch.int64)
            or input_ids.shape[0] == 0 or input_ids.shape[1] == 0):
        return None
    sequence = input_ids.shape[1]
    binary = ((attention_mask == 0) | (attention_mask == 1)).all()
    valid = attention_mask != 0
    counts = valid.sum(dim=-1, dtype=torch.int32)
    prefix = torch.arange(sequence, device=input_ids.device)[None, :] < counts[:, None]
    no_removed_labels = (valid | (labels == -100)).all()
    supported = (binary & (valid == prefix).all() & (counts > 0).all()
                 & no_removed_labels & (counts.sum() >= sequence))
    if not bool(supported):
        return None
    zero = counts.new_zeros((1,))
    cumulative = torch.cat((zero, counts.cumsum(dim=0, dtype=torch.int32)))
    return {"cumulative": cumulative, "maximum": sequence}


# ---------------------------------------------------------------- the forward

class _Batch:
    """Per-step state: the real-token rows and the padded copies LoRA dropout needs."""

    def __init__(self, rows, batch, seq_len, packed_metadata=None):
        self.rows, self.batch, self.seq_len = rows, batch, seq_len
        self.packed_metadata = packed_metadata
        self._padded = {}

    def dropout(self, dropout, x):
        """dropout(x) of the real tokens, drawn on the (batch, seq_len, features) tensor of the
        base run, so the random masks are the same."""
        if type(dropout) is torch.nn.Identity:
            return x
        # self.rows is the matching attention_mask.reshape(-1).nonzero()
        # producer, hence unique and in range. Native dropout still draws on
        # the identical full shape; only real values/masks enter autograd.
        if (REAL_ROW_DROPOUT
                and self.batch * self.seq_len * x.shape[-1] >= REAL_ROW_DROPOUT_MIN_PADDED_ELEMENTS
                and type(dropout) is torch.nn.Dropout
                and dropout.training is True and dropout.inplace is False
                and torch.nn.Dropout.forward is _NATIVE_DROPOUT_FORWARD
                and getattr(dropout.forward, "__func__", None) is _NATIVE_DROPOUT_FORWARD
                and getattr(_NATIVE_DROPOUT_FORWARD, "__module__", None) == "torch.nn.modules.dropout"
                and getattr(_NATIVE_DROPOUT_FORWARD, "__qualname__", None) == "Dropout.forward"
                and not hasattr(_NATIVE_DROPOUT_FORWARD, "__wrapped__")
                and not hasattr(dropout, "_hf_hook")
                and not any(getattr(dropout, name, None) for name in (
                    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"))
                and not any(getattr(torch.nn.modules.module, name, None) for name in (
                    "_global_forward_hooks", "_global_forward_pre_hooks",
                    "_global_backward_hooks", "_global_backward_pre_hooks"))
                and str(torch.__version__).split("+", 1)[0] == "2.11.0"
                and _real_dropout_supported(x, self.rows, self.batch, self.seq_len,
                                             dropout.p, True, True)):
            return real_row_dropout(x, self.rows, self.batch, self.seq_len,
                                    dropout.p, rows_from_nonzero=True)
        key = id(x)
        if key not in self._padded:
            full = x.new_zeros((self.batch * self.seq_len, x.shape[-1]))
            self._padded[key] = (x, full.index_copy(0, self.rows, x).view(
                self.batch, self.seq_len, x.shape[-1]))
        padded = self._padded[key][1]
        return dropout(padded).view(self.batch * self.seq_len, -1).index_select(0, self.rows)

    def done_with(self, x):
        self._padded.pop(id(x), None)


def _lora_out(parts, x, step):
    base, lora_a, lora_b, dropout, scaling = parts
    if lora_a is None:
        return None
    return lora_b(lora_a(step.dropout(dropout, x))) * scaling


def _attention(layer, projections, x, cos, sin, mask, step, meta):
    attn = layer.self_attn
    q_parts, k_parts, v_parts, o_parts = projections
    q, k, v = QKV4bit.apply(x, q_parts[0], k_parts[0], v_parts[0], torch.float16)
    # the LoRA branches in the base order (q, k, v): same dropout draws
    lq = _lora_out(q_parts, x, step)
    lk = _lora_out(k_parts, x, step)
    lv = _lora_out(v_parts, x, step)
    step.done_with(x)
    if step.packed_metadata is not None:
        # Equal heads keep native efficient/Cutlass available on sm75. No
        # PyTorch repeat/gather is necessary: the Triton producer writes this
        # packed layout directly, in the same real-token order as x.
        qp, kp, vp = RoPEPacked.apply(q, lq, k, lk, v, lv, cos, sin, step.rows,
            step.batch, step.seq_len, meta["hq"], meta["hk"], meta["dim"], meta["rep"])
        cumulative = step.packed_metadata["cumulative"]
        maximum = step.packed_metadata["maximum"]
        out = torch.ops.aten._efficient_attention_forward.default(
            qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2), None,
            cumulative, cumulative, maximum, maximum, 0.0, 1, True,
            scale=attn.scaling)[0]
        a = out.reshape(step.rows.numel(), meta["hq"] * meta["dim"])
        del qp, kp, vp, out
    else:
        # Legacy helper path retained for direct component comparisons.
        use_gqa = mask is None and meta["rep"] > 1
        rep = 1 if use_gqa else meta["rep"]
        qp, kp, vp = RoPEPad.apply(q, lq, k, lk, v, lv, cos, sin, step.rows, step.batch,
                                   step.seq_len, meta["hq"], meta["hk"], meta["dim"], rep)
        is_causal = mask is None and step.seq_len > 1 and getattr(attn, "is_causal", True)
        kwargs = {"enable_gqa": True} if use_gqa else {}
        out = torch.nn.functional.scaled_dot_product_attention(
            qp, kp, vp, attn_mask=mask, dropout_p=0.0, scale=attn.scaling,
            is_causal=is_causal, **kwargs)
        del qp, kp, vp
        a = UnpadHeads.apply(out, step.rows, step.seq_len)
    o = MatMulNF4.apply(a, o_parts[0].weight, None, o_parts[0].weight.quant_state)
    lo = _lora_out(o_parts, a, step)
    step.done_with(a)
    return o if lo is None else o + lo


def fast_forward(model, meta, input_ids, attention_mask, labels, num_items_in_batch,
                 packed_metadata=None):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if packed_metadata is None:
        packed_metadata = _batch_metadata(input_ids, attention_mask, labels)
        if packed_metadata is None:
            raise ValueError("unsupported packed right-padded batch")
    patch_metric_readers()
    _warmup.wait()  # a kernel still compiling in the warm-up thread: wait, do not compile twice
    inner = model.model
    batch, seq_len = input_ids.shape
    rows = attention_mask.reshape(-1).nonzero().squeeze(1)
    step = _Batch(rows, batch, seq_len, packed_metadata=packed_metadata)

    h = inner.embed_tokens(input_ids.reshape(-1).index_select(0, rows))
    cos, sin = inner.rotary_emb(h, (rows % seq_len)[None])  # the positions of the base run
    cos, sin = cos[0].contiguous(), sin[0].contiguous()
    # Cumulative row boundaries and custom_mask_type1 provide causal isolation;
    # no dense [B,1,S,S] causal/padding mask is needed in this private path.
    mask = None

    mlp_out = None
    for layer, projections in meta["layers"]:
        norm = layer.input_layernorm
        if mlp_out is None:  # layer 0: h + 0 is h; the same kernel as the other norms
            mlp_out = h.new_zeros(h.shape, dtype=torch.float16)
        h, x = add_rmsnorm(h, mlp_out, norm.weight, norm.variance_epsilon)
        o = _attention(layer, projections, x, cos, sin, mask, step, meta)
        norm = layer.post_attention_layernorm
        h, x = add_rmsnorm(h, o, norm.weight, norm.variance_epsilon)
        mlp = layer.mlp
        mlp_out = MLP4bit.apply(x, mlp.gate_proj, mlp.up_proj, mlp.down_proj, torch.float16)
    norm = inner.norm
    _, x = add_rmsnorm(h, mlp_out, norm.weight, norm.variance_epsilon)

    # ForCausalLMLoss: the label of position t is labels[t + 1]; -100 past the end
    shifted = torch.cat([labels[:, 1:], labels.new_full((batch, 1), -100)], dim=1)
    targets = shifted.reshape(-1).index_select(0, rows)
    loss, stats = lm_head_loss(x, model.lm_head.weight, targets, num_items_in_batch)
    logits = LazyLogits(batch, seq_len, model.lm_head.weight.shape[0], rows, stats)
    return CausalLMOutputWithPast(loss=loss, logits=logits)


_ALLOWED_KWARGS = {"num_items_in_batch", "output_attentions", "output_hidden_states",
                   "return_dict"}


def make_causal_lm_forward(original, *, private_only=False):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        eligible = (
            self.training and torch.is_grad_enabled() and input_ids is not None
            and inputs_embeds is None and labels is not None and attention_mask is not None
            and position_ids is None and past_key_values is None and not use_cache
            and isinstance(logits_to_keep, int) and logits_to_keep == 0
            and input_ids.is_cuda and input_ids.dim() == 2
            and attention_mask.shape == input_ids.shape and labels.shape == input_ids.shape
            and set(kwargs) <= _ALLOWED_KWARGS
            and not kwargs.get("output_attentions") and not kwargs.get("output_hidden_states")
            and kwargs.get("return_dict", True) is not False
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
        if eligible and private_only:
            from .private_scope import permits
            eligible = permits(self, input_ids, labels, attention_mask)
        meta = _layout(self) if eligible else None
        packed_metadata = _batch_metadata(input_ids, attention_mask, labels) if meta is not None else None
        if packed_metadata is None:
            meta = None
        if meta is not None:
            try:
                return fast_forward(self, meta, input_ids, attention_mask, labels,
                                    kwargs.get("num_items_in_batch"), packed_metadata=packed_metadata)
            except Exception as exc:
                # Replaying the original after a partial fast forward could
                # consume dropout RNG twice. Unsupported calls fall back above;
                # implementation errors must be visible instead of replayed.
                raise RuntimeError("private unpadded forward failed; not replaying dropout") from exc
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._soup_private_unpad = private_only
    return forward


def patch_qwen2_causal_lm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    cls.forward = make_causal_lm_forward(cls.forward, private_only=True)


def warmup(parts=("fwd", "bwd"), hq=12, hk=2, dim=128):
    """Compile the kernels of this file for Qwen2.5-1.5B's training step; see warmup.py."""
    f16, f32, i64 = torch.float16, torch.float32, torch.int64
    bq, bk = _blocks(hq, hk)
    n = 16  # every stride of these layouts is a multiple of 16; seq_len is not specialized
    common = dict(HQ=hq, HK=hk, HALF=dim // 2, REP=hq // hk, BLOCK_HQ=bq, BLOCK_HK=bk,
                  PACKED=True,
                  HAS_LQ=True, HAS_LK=False, HAS_LV=True, num_warps=4, enable_fp_fusion=False)
    if "fwd" in parts:
        rope_pad_fwd_kernel.warmup(f16, f16, f16, f16, f16, f16, f32, f32, i64, f16, f16, f16,
                                   n, n, n, n, n, n, 7, grid=(1,), **common)
    if "bwd" in parts:
        rope_pad_bwd_kernel.warmup(f16, f16, f16, f32, f32, i64, f16, f16, f16, f16, f16, f16,
                                   n, n, n, n, n, n, n, n, n, 7, grid=(1,), **common)

"""Parallel vocabulary tiles for the shared team's chunked LM head.

Nurzhigit/Codex implementation extending the v10 tiled reduction. The shared
head and unpadding remain Vladimir's contribution (see TEAM_INTEGRATION.md).
"""
import torch
import triton
import triton.language as tl

TILE = 4096


@triton.jit
def head_tile_statistics(X, PM, PS, PW, PI, V: tl.constexpr,
                         STRIDE: tl.constexpr, PARTS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * STRIDE + col, col < V, other=-float('inf')).to(tl.float32)
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    local_argmax = tl.min(tl.where(z == m, col, 2147483647), 0)
    location = row * PARTS + part
    tl.store(PM + location, m)
    tl.store(PS + location, tl.sum(e, 0))
    tl.store(PW + location, tl.sum(e * tl.where(col < V, z, 0.0), 0))
    tl.store(PI + location, local_argmax)


@triton.jit
def head_finish_statistics(X, TARGET, PM, PS, PW, PI, LSE, ENT, AMAX, CE,
                           V: tl.constexpr, STRIDE: tl.constexpr,
                           PARTS: tl.constexpr, WITH_GRAD: tl.constexpr,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    m = tl.load(PM + row * PARTS + part, part < PARTS, other=-float('inf'))
    s = tl.load(PS + row * PARTS + part, part < PARTS, other=0.0)
    w = tl.load(PW + row * PARTS + part, part < PARTS, other=0.0)
    i = tl.load(PI + row * PARTS + part, part < PARTS, other=2147483647)
    largest = tl.max(m, 0)
    factor = tl.exp(m - largest)
    total = tl.sum(factor * s, 0)
    log_total = largest + tl.log(total)
    best = tl.min(tl.where(m == largest, i, 2147483647), 0)
    tl.store(LSE + row, log_total)
    tl.store(ENT + row, log_total - tl.sum(factor * w, 0) / total)
    tl.store(AMAX + row, best.to(tl.int64))
    if WITH_GRAD:
        target = tl.load(TARGET + row)
        selected = tl.load(X + row * STRIDE + target).to(tl.float32)
        tl.store(CE + row, log_total - selected)


@triton.jit
def head_gradient_inplace(X, TARGET, LSE, V: tl.constexpr,
                          STRIDE: tl.constexpr, SCALE: tl.constexpr,
                          BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * STRIDE + col, col < V, other=0.0).to(tl.float32)
    label = tl.load(TARGET + row)
    normalizer = tl.load(LSE + row)
    grad = (tl.exp(z - normalizer) - (col == label).to(tl.float32)) * SCALE
    tl.store(X + row * STRIDE + col, grad, col < V)


def reduce_chunk(logits, targets=None, scale=1024.0):
    """Return entropy/argmax/CE and optionally replace logits by scaled dCE/dz."""
    n, vocab = logits.shape
    parts = triton.cdiv(vocab, TILE)
    pm = torch.empty((n, parts), device=logits.device, dtype=torch.float32)
    ps, pw = torch.empty_like(pm), torch.empty_like(pm)
    pi = torch.empty((n, parts), device=logits.device, dtype=torch.int32)
    lse = torch.empty(n, device=logits.device, dtype=torch.float32)
    entropy, ce = torch.empty_like(lse), torch.empty_like(lse)
    argmax = torch.empty(n, device=logits.device, dtype=torch.int64)
    head_tile_statistics[(n, parts)](logits, pm, ps, pw, pi, vocab,
                                    logits.stride(0), parts, TILE, num_warps=4)
    head_finish_statistics[(n,)](logits, targets if targets is not None else pi,
                                pm, ps, pw, pi, lse, entropy, argmax, ce,
                                vocab, logits.stride(0), parts, targets is not None,
                                triton.next_power_of_2(parts), num_warps=4)
    if targets is not None:
        head_gradient_inplace[(n, parts)](logits, targets, lse, vocab,
                                          logits.stride(0), scale, TILE, num_warps=4)
    return entropy, argmax, ce


def parallel_stats(x16, w16, idx, targets, ent, amax, ce, grad_x):
    from .team_vladimir.lmhead import CHUNK, GRAD_SCALE
    if idx.numel() == 0:
        return
    for chunk in idx.split(CHUNK):
        logits = torch.mm(x16.index_select(0, chunk), w16.t())
        target = targets.index_select(0, chunk) if targets is not None else None
        e, a, c = reduce_chunk(logits, target, GRAD_SCALE)
        ent.index_copy_(0, chunk, e)
        amax.index_copy_(0, chunk, a)
        if targets is not None:
            ce.index_copy_(0, chunk, c)
            grad_x.index_copy_(0, chunk, torch.mm(logits, w16).float())


def patch_parallel_head():
    from .team_vladimir import lmhead
    lmhead._stats = parallel_stats
    # Registered before apply_team starts its compile thread. Compile the
    # kernels this head actually uses, rather than the superseded serial loop.
    lmhead.warmup = warmup


def warmup(vocab=151936):
    """Compile current signatures inside this run while Soup loads the model.

    These calls do not launch training work or load a cache from another run.
    A failed warmup is handled by the existing team thread: normal first launch
    compiles that signature. All ordinary training launches remain unchanged.
    """
    from .compact_dropout import warmup as warmup_dropout
    from .frozen_storage_kernel import warmup as warmup_storage
    f16, f32, i32, i64 = torch.float16, torch.float32, torch.int32, torch.int64
    parts = triton.cdiv(vocab, TILE)
    head_tile_statistics.warmup(f16, f32, f32, f32, i32, vocab, vocab,
                                parts, TILE, grid=(1, parts), num_warps=4)
    for with_grad in (False, True):
        head_finish_statistics.warmup(
            f16, i64 if with_grad else i32, f32, f32, f32, i32,
            f32, f32, i64, f32, vocab, vocab, parts, with_grad,
            triton.next_power_of_2(parts), grid=(1,), num_warps=4)
    head_gradient_inplace.warmup(f16, i64, f32, vocab, vocab, 1024.0,
                                 TILE, grid=(1, parts), num_warps=4)
    warmup_dropout()
    warmup_storage()

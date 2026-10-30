"""Chunked frozen lm_head CE/metrics with actual-scale backward.

The online statistics and private LazyLogits interface originate from teammate
Vladimir's implementation at main 987da36e1efa90652e1a7f8c48afef50685dbc40.
The team correction applies the true grad_loss/denominator BEFORE FP16
gradient rounding and the half head-gradient GEMM. In the strict private
trainer scope, its live GradScaler value allows this work to run during
forward; only half hidden states/gradients and FP32 row LSE are retained.
Backward checks the actual upstream value. A mismatch recomputes labelled
head logits and uses the actual value, preserving arbitrary upstream scaling.

General calls retain the qualified supervised FP16 logits/LSE cache path.
No fixed 1024 gradient is used. LazyLogits must remain inside the caller's
strict private trainer scope; its interface is unchanged.
Online softmax idea: https://arxiv.org/abs/1805.02867.
"""

import torch
import triton
import triton.language as tl

CHUNK = 512
BLOCK_V = 4096
GRAD_BLOCK = 1024


@triton.jit
def _lm_ce_statistics(LOGITS, target, stride, V, BLOCK_V: tl.constexpr,
                       WITH_TARGET: tl.constexpr):
    row = tl.program_id(0)
    ptr = LOGITS + row.to(tl.int64) * stride
    m = -1e30
    s = 0.0
    u = 0.0
    best = -1e30
    best_i = 0
    zt = 0.0
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.where(mask, z, -1e30)
        bmax = tl.max(z, axis=0)
        bi = tl.argmax(z, axis=0)
        best_i = tl.where(bmax > best, start + bi, best_i)
        best = tl.maximum(best, bmax)
        new_m = tl.maximum(m, bmax)
        alpha = tl.exp(m - new_m)
        e = tl.exp(z - new_m)
        u = alpha * (u + s * (m - new_m)) + tl.sum(e * (z - new_m), axis=0)
        s = alpha * s + tl.sum(e, axis=0)
        m = new_m
        if WITH_TARGET:
            zt += tl.sum(tl.where(cols == target, z, 0.0), axis=0)
    log_s = tl.log(s)
    lse = m + log_s
    return log_s - u / s, best_i.to(tl.int64), lse - zt, lse


@triton.jit
def lm_ce_kernel(LOGITS, TARGET, ENT, AMAX, CE, LSE, stride, V,
                 BLOCK_V: tl.constexpr):
    """Dense-output compatibility entry point, sharing the production arithmetic."""
    row = tl.program_id(0)
    target = tl.load(TARGET + row).to(tl.int32)
    entropy, best, ce, lse = _lm_ce_statistics(
        LOGITS, target, stride, V, BLOCK_V, True)
    tl.store(ENT + row, entropy)
    tl.store(AMAX + row, best)
    tl.store(CE + row, ce)
    tl.store(LSE + row, lse)


@triton.jit
def lm_ce_scatter_kernel(LOGITS, TARGET, ROWS, ENT, AMAX, CE, LSE,
                         stride, row_stride, target_stride, V,
                         BLOCK_V: tl.constexpr):
    """One signature for all rows; ignored targets never write CE/LSE storage."""
    row = tl.program_id(0)
    output_row = tl.load(ROWS + row.to(tl.int64) * row_stride)
    target = tl.load(TARGET + output_row * target_stride).to(tl.int32)
    entropy, best, ce, lse = _lm_ce_statistics(
        LOGITS, target, stride, V, BLOCK_V, True)
    tl.store(ENT + output_row, entropy)
    tl.store(AMAX + output_row, best)
    labelled = target != -100
    tl.store(CE + output_row, ce, mask=labelled)
    tl.store(LSE + row, lse, mask=labelled)


@triton.jit
def lm_scale_gradient_kernel(LOGITS, LSE, TARGET, SCALE, DG, stride,
                              V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < V
    z = tl.load(LOGITS + row.to(tl.int64) * stride + col,
                 mask=mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row).to(tl.float32)
    target = tl.load(TARGET + row)
    scale = tl.load(SCALE).to(tl.float32)
    probability = tl.exp(z - lse)
    gradient = probability - (col == target).to(tl.float32)
    scaled = gradient * scale
    if tl.abs(scale) == float("inf"):
        # Native NLL -> log-softmax gives -scale + probability * scale:
        # the target is NaN for either sign of infinite scale, also at p=0.
        scaled = tl.where(col == target, float("nan"), scaled)
    tl.store(DG + row.to(tl.int64) * V + col,
              scaled.to(DG.dtype.element_ty), mask=mask)


def _targets_in_bounds(targets, vocabulary):
    """Raw flag is testable without poisoning a CUDA process with an assert."""
    return ((targets == -100) | ((targets >= 0) & (targets < vocabulary))).all()


def _private_expected_loss_scale():
    # Resolve lazily: importing this module alone never activates a trainer
    # scope or depends on construction order during setup/warmup.
    from . import private_scope
    helper = getattr(private_scope, "expected_loss_scale", None)
    return helper() if helper is not None else None


def _gradient_mm(logits, lse, targets, scale, weight, overwrite=False):
    vocabulary = weight.shape[0]
    dg = logits if overwrite else torch.empty_like(logits)
    lm_scale_gradient_kernel[(logits.shape[0], triton.cdiv(vocabulary, GRAD_BLOCK))](
        logits, lse, targets, scale, dg, logits.stride(0), vocabulary,
        BLOCK=GRAD_BLOCK, num_warps=4)
    return torch.mm(dg, weight)


def _stats(x16, w16, idx, targets, entropy, argmax, ce, saved_chunks=None,
           precompute=None):
    if idx.numel() == 0:
        return
    vocabulary = w16.shape[0]
    for chunk in idx.split(CHUNK):
        count = chunk.numel()
        logits = torch.mm(x16.index_select(0, chunk), w16.t())
        # Each index occurs once. Scatter the statistics in the same launch
        # that computes them, avoiding chunk temporaries and index_copy calls.
        # Prompt rows use the same compiled kernel, whose ignored-target mask
        # leaves CE/LSE untouched. Their unused pointers may alias entropy.
        lse = (torch.empty(count, device=x16.device, dtype=torch.float32)
               if ce is not None else None)
        lm_ce_scatter_kernel[(count,)](
            logits, targets, chunk, entropy, argmax,
            ce if ce is not None else entropy,
            lse if lse is not None else entropy,
            logits.stride(0), chunk.stride(0), targets.stride(0), vocabulary,
            BLOCK_V=BLOCK_V, num_warps=8)
        if saved_chunks is not None:
            # Keep the exact original half GEMM output. No clone/index_copy of
            # this vocabulary-sized tensor and no in-place gradient overwrite.
            saved_chunks.append((logits, lse))
        if precompute is not None:
            scale, hidden_gradient, saved_lse = precompute
            # Statistics have consumed logits. Reuse that temporary as half
            # dlogits rather than retaining/allocating a vocabulary-sized cache.
            target = targets.index_select(0, chunk)
            dx = _gradient_mm(logits, lse, target, scale, w16, overwrite=True)
            hidden_gradient.index_copy_(0, chunk, dx)
            saved_lse.append(lse)
            del dx, target
        # Release the previous chunk BEFORE the next GEMM allocates its output.
        # Private/precomputed and prompt chunks have no later consumer. General
        # autograd retains its exact original tensor through saved_chunks.
        del logits


class LMHeadLoss(torch.autograd.Function):
    """Same scalar CE/stats API, with true upstream scaling before half DG."""

    @staticmethod
    def forward(ctx, x, weight, targets, denom, stats):
        if weight.requires_grad:
            raise ValueError("chunked lm_head requires frozen weight")
        if (x.ndim != 2 or weight.ndim != 2 or targets.ndim != 1
                or x.shape[1] != weight.shape[1] or targets.numel() != x.shape[0]
                or targets.dtype != torch.int64 or targets.device != x.device
                or weight.device != x.device or not x.is_cuda):
            raise ValueError("unsupported chunked lm_head input shape/device/dtype")
        torch._assert_async(_targets_in_bounds(targets, weight.shape[0]),
                            "cross_entropy target out of bounds")
        tokens, hidden = x.shape
        x16 = x.to(torch.float16)
        expected = (_private_expected_loss_scale()
                    if ctx.needs_input_grad[0] and x.dtype == torch.float32 else None)
        if expected is not None:
            # Only the already-qualified private training scope reuses this
            # exact cast. Public/general calls keep their ordinary lifetime.
            from .head_weight_cache import get_half_weight
            w16 = get_half_weight(weight)
        else:
            w16 = weight.to(torch.float16)
        entropy = torch.empty(tokens, device=x.device, dtype=torch.float32)
        argmax = torch.empty(tokens, device=x.device, dtype=torch.int64)
        ce = torch.zeros(tokens, device=x.device, dtype=torch.float32)
        valid = targets != -100
        labelled = valid.nonzero().squeeze(1)
        prompt = (~valid).nonzero().squeeze(1)
        chunks = []
        ctx.precomputed = expected is not None
        if ctx.precomputed:
            expected = torch.tensor(expected, device=x.device, dtype=torch.float32)
            hidden_gradient = torch.zeros((tokens, hidden), device=x.device,
                                           dtype=torch.float16)
            lse_chunks = []
            precompute = (expected / denom.float(), hidden_gradient, lse_chunks)
        else:
            precompute = None
        _stats(x16, w16, labelled, targets, entropy, argmax, ce,
               chunks if ctx.needs_input_grad[0] and not ctx.precomputed else None,
               precompute=precompute)
        _stats(x16, w16, prompt, targets, entropy, argmax, None)
        stats.update(entropy=entropy, argmax=argmax, ce=ce, valid=valid)
        labelled_targets = targets.index_select(0, labelled)
        if ctx.precomputed:
            # FP32 x guarantees x16 is an owned copy, so a later mutation of
            # the public input cannot corrupt the mismatch recomputation.
            ctx.num_chunks = len(lse_chunks)
            ctx.save_for_backward(w16, labelled, labelled_targets, denom,
                                  x16, expected, hidden_gradient, *lse_chunks)
        else:
            ctx.num_chunks = len(chunks)
            ctx.save_for_backward(w16, labelled, labelled_targets, denom,
                                  *(value[0] for value in chunks),
                                  *(value[1] for value in chunks))
        ctx.x_dtype, ctx.shape = x.dtype, (tokens, hidden)
        return ce.sum() / denom

    @staticmethod
    def backward(ctx, grad_loss):
        w16, labelled, labelled_targets, denom, *buffers = ctx.saved_tensors
        # Ignore-index derivatives stay zero even at NaN/Inf loss/upstream.
        if labelled.numel() == 0:
            return (torch.zeros(ctx.shape, device=w16.device, dtype=ctx.x_dtype),
                    None, None, None, None)
        if ctx.precomputed:
            x16, expected, hidden_gradient, *lse_chunks = buffers
            # One scalar equality, without assuming the caller used its
            # predicted AMP scale. Mismatches take the actual-scale path.
            if torch.equal(grad_loss.detach().float().reshape(()), expected):
                return hidden_gradient.to(ctx.x_dtype), None, None, None, None
            logits_chunks = None
        else:
            logits_chunks = buffers[:ctx.num_chunks]
            lse_chunks = buffers[ctx.num_chunks:]
        grad_x = torch.zeros(ctx.shape, device=w16.device, dtype=ctx.x_dtype)
        scale = grad_loss.float() / denom.float()
        offset = 0
        for position, lse in enumerate(lse_chunks):
            count = lse.shape[0]
            indices = labelled[offset:offset + count]
            if ctx.precomputed:
                logits = torch.mm(x16.index_select(0, indices), w16.t())
            else:
                logits = logits_chunks[position]
            target_chunk = labelled_targets[offset:offset + count]
            dx = _gradient_mm(logits, lse, target_chunk, scale, w16,
                              overwrite=ctx.precomputed)
            grad_x.index_copy_(0, indices, dx.to(ctx.x_dtype))
            offset += count
            # The scale-mismatch path recomputes each chunk. Do not overlap
            # its dead logits with the next one. General saved graphs still
            # own their original logits through ctx.saved_tensors.
            del logits, dx
        return grad_x, None, None, None, None


def lm_head_loss(x, weight, targets, num_items_in_batch):
    if num_items_in_batch is None:
        denom = (targets != -100).sum().float()
    elif torch.is_tensor(num_items_in_batch):
        denom = num_items_in_batch.to(x.device)
    else:
        denom = torch.tensor(num_items_in_batch, device=x.device)
    stats = {}
    loss = LMHeadLoss.apply(x, weight, targets, denom, stats)
    return loss, stats


class LazyLogits:
    """Stands in for the (batch, seq_len, vocab) logits in the model output.

    The full logits never exist in this run. The three readers of `outputs.logits` in a training
    step get the per-token values they compute, through the patches below:
    TRL's entropy_from_logits (entropy per position), TRL's token accuracy
    (`logits[..., :-1, :].contiguous().argmax(dim=-1)`) and Soup's rewind row_losses. Anything
    else raises, so a new reader cannot silently get wrong values.
    """

    def __init__(self, batch, seq_len, vocab, rows, stats):
        self.shape = torch.Size((batch, seq_len, vocab))
        self._rows, self._stats = rows, stats

    def _full(self, values, fill):
        batch, seq_len, _ = self.shape
        out = values.new_full((batch * seq_len,), fill)
        out.index_copy_(0, self._rows, values)
        return out.view(batch, seq_len)

    def entropy(self):
        return self._full(self._stats["entropy"], 0.0)

    def predictions(self):
        return self._full(self._stats["argmax"], -1)

    def token_ce(self):
        """Cross-entropy of position t predicting label t + 1, 0 where that label is -100."""
        return self._full(self._stats["ce"], 0.0)

    def __getitem__(self, key):
        if key == (Ellipsis, slice(None, -1, None), slice(None, None, None)):
            return _ShiftedLazyLogits(self)
        raise TypeError(f"LazyLogits does not support indexing with {key!r}")

    def __getattr__(self, name):
        raise AttributeError(f"LazyLogits has no {name!r}: the full logits are not materialized")


class _ShiftedLazyLogits:
    """`logits[..., :-1, :]`, as TRL's token accuracy reads it."""

    def __init__(self, parent):
        self._parent = parent

    def contiguous(self):
        return self

    def argmax(self, dim=-1):
        if dim not in (-1, 2):
            raise TypeError("only the argmax over the vocabulary is available")
        return self._parent.predictions()[:, :-1]

    def __getattr__(self, name):
        raise AttributeError(f"shifted LazyLogits has no {name!r}")


def make_entropy_from_logits(original):
    def entropy_from_logits(logits, chunk_size=128):
        if isinstance(logits, LazyLogits):
            return logits.entropy()
        return original(logits, chunk_size)

    return entropy_from_logits


def make_row_losses(original):
    def row_losses(logits, labels):
        if not isinstance(logits, LazyLogits):
            return original(logits, labels)
        # Soup's row_losses: per row, the mean over labels[:, 1:] != -100 of the token CE
        with torch.no_grad():
            mask = labels[:, 1:] != -100
            token_ce = logits.token_ce()[:, :-1].to(labels.device)
            tokens = mask.sum(dim=1)
            totals = (token_ce * mask.to(token_ce.dtype)).sum(dim=1)
            means = totals / tokens.clamp(min=1).to(totals.dtype)
        return [float(v) for v in means], [int(v) for v in tokens]

    return row_losses


def patch_metric_readers():
    """Point TRL's entropy and Soup's rewind row losses at LazyLogits. Done when the fast forward
    first runs, from the modules already loaded: nothing is imported early, so Soup's tests that
    swap in fake modules are unaffected."""
    import sys

    sft = sys.modules.get("trl.trainer.sft_trainer")
    if sft is not None and not getattr(sft.entropy_from_logits, "_solution", False):
        sft.entropy_from_logits = make_entropy_from_logits(sft.entropy_from_logits)
        sft.entropy_from_logits._solution = True
    rewind = sys.modules.get("soup_cli.trainer.rewind_hf")
    if rewind is not None and not getattr(rewind.row_losses, "_solution", False):
        rewind.row_losses = make_row_losses(rewind.row_losses)
        rewind.row_losses._solution = True


def warmup(vocab=151936):
    """Compile the production scatter statistics and actual-scale backward."""
    from .warmup import load_compiled

    f16, f32, i64 = torch.float16, torch.float32, torch.int64
    load_compiled(lm_ce_scatter_kernel.warmup(
        f16, i64, i64, f32, i64, f32, f32, vocab, 1, 1, vocab,
        grid=(1,), BLOCK_V=BLOCK_V, num_warps=8))
    load_compiled(lm_scale_gradient_kernel.warmup(f16, f32, i64, f32, f16, vocab, vocab,
                                    grid=(1, 1), BLOCK=GRAD_BLOCK, num_warps=4))

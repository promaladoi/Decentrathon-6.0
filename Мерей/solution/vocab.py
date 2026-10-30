"""Original tiled vocabulary reductions; no dense softmax/log-softmax workspace.

The vocabulary is split across CTAs so a 151936-class row never occupies one
enormous Triton block. Partial maxima, masses and centered moments are combined
with the log-sum-exp identity. Backward recomputes probabilities from the logits.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

TILE = 4096


@triton.jit
def vocabulary_partials_kernel(X, Partial, V: tl.constexpr, PARTS: tl.constexpr,
                               ROWS, ENTROPY: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * V + col, col < V, other=-float("inf"))
    m = tl.max(z, 0)
    # A completely masked (-inf) partition contributes zero mass to CE,
    # including when other partitions in the same row have finite values.
    centered = tl.where(m == -float("inf"), -float("inf"), z - m)
    p = tl.exp(centered)
    mass = tl.sum(p, 0)
    index = row * PARTS + part
    plane = ROWS * PARTS
    tl.store(Partial + index, m)
    tl.store(Partial + plane + index, mass)
    if ENTROPY:
        moment = tl.sum(tl.where(col < V, p * centered, 0.0), 0)
        tl.store(Partial + 2 * plane + index, moment)


@triton.jit
def vocabulary_finish_kernel(X, Targets, Partial, Norm, Output,
                             V: tl.constexpr, PARTS: tl.constexpr,
                             ROWS, IGNORE: tl.constexpr,
                             ENTROPY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    plane = ROWS * PARTS
    index = row * PARTS + part
    m = tl.load(Partial + index, part < PARTS, other=-float("inf"))
    s = tl.load(Partial + plane + index, part < PARTS, other=0.0)
    largest = tl.max(m, 0)
    factor = tl.exp(m - largest)
    total = tl.sum(s * factor, 0)
    log_total = tl.log(total)
    if ENTROPY:
        t = tl.load(Partial + 2 * plane + index, part < PARTS, other=0.0)
        shifted = tl.where(part < PARTS, t + s * (m - largest), 0.0)
        moment = tl.sum(factor * shifted, 0)
        tl.store(Output + row, log_total - moment / total)
    else:
        target = tl.load(Targets + row)
        valid = (target != IGNORE) & (target >= 0) & (target < V)
        selected = tl.load(X + row * V + target, valid, other=0.0)
        tl.store(Output + row, tl.where(valid, (largest - selected) + log_total, 0.0))
        tl.store(Norm + row * 2, largest)
        tl.store(Norm + row * 2 + 1, total)


@triton.jit
def vocabulary_gradient_kernel(X, Targets, Norm, DOut, Count, DX,
                               V: tl.constexpr, IGNORE: tl.constexpr,
                               REDUCTION: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * V + col, col < V, other=0.0)
    m = tl.load(Norm + 2 * row)
    mass = tl.load(Norm + 2 * row + 1)
    target = tl.load(Targets + row)
    if REDUCTION == 0:
        scale = tl.load(DOut + row)
    else:
        scale = tl.load(DOut)
        if REDUCTION == 2:
            count = tl.load(Count).to(tl.float32)
            scale = scale / tl.maximum(count, 1.0)
    gradient = (tl.exp(z - m) / mass - (col == target).to(tl.float32)) * scale
    gradient = tl.where(target != IGNORE, gradient, 0.0)
    tl.store(DX + row * V + col, gradient, col < V)


def _statistics(logits, targets=None, ignore_index=-100):
    rows, vocab = logits.shape
    entropy = targets is None
    parts = triton.cdiv(vocab, TILE)
    partial = torch.empty((3 if entropy else 2, rows, parts),
                          device=logits.device, dtype=torch.float32)
    norm = torch.empty((rows, 2), device=logits.device, dtype=torch.float32)
    output = torch.empty(rows, device=logits.device, dtype=torch.float32)
    vocabulary_partials_kernel[(rows, parts)](
        logits, partial, vocab, parts, rows, entropy, TILE, num_warps=4)
    vocabulary_finish_kernel[(rows,)](
        logits, targets if targets is not None else output, partial, norm, output,
        vocab, parts, rows, ignore_index, entropy, triton.next_power_of_2(parts),
        num_warps=4)
    return output, norm


class _VocabularyCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index, reduction):
        # Also protects the masked gather and backward against invalid targets.
        torch._assert_async(((targets == ignore_index) |
                             ((targets >= 0) & (targets < logits.shape[1]))).all(),
                            "cross_entropy target out of bounds")
        losses, norm = _statistics(logits, targets, ignore_index)
        count = (targets != ignore_index).sum()
        ctx.save_for_backward(logits, targets, norm, count)
        ctx.ignore_index, ctx.reduction = ignore_index, reduction
        if reduction == "none":
            return losses
        total = losses.sum()
        return total if reduction == "sum" else total / count

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, targets, norm, count = ctx.saved_tensors
        grad = torch.empty_like(logits)
        rows, vocab = logits.shape
        vocabulary_gradient_kernel[(rows, triton.cdiv(vocab, TILE))](
            logits, targets, norm, grad_output.contiguous(), count, grad, vocab,
            ctx.ignore_index, {"none": 0, "sum": 1, "mean": 2}[ctx.reduction],
            TILE, num_warps=4)
        return grad, None, None, None


def cross_entropy(logits, targets, ignore_index=-100, reduction="mean"):
    """Float32, contiguous [tokens, vocabulary], hard int64 targets only."""
    from .patches import _can_ce

    if not _can_ce(logits, targets) or reduction not in ("none", "sum", "mean"):
        return torch.nn.functional.cross_entropy(logits, targets,
                                                ignore_index=ignore_index,
                                                reduction=reduction)
    return _VocabularyCrossEntropy.apply(logits, targets, ignore_index, reduction)


def entropy_from_logits(logits):
    """No-grad entropy, retaining exactly the leading dimensions of logits."""
    from .patches import _can_reduce

    if not _can_reduce(logits) or (torch.is_grad_enabled() and logits.requires_grad):
        probabilities = torch.softmax(logits, dim=-1)
        return logits.logsumexp(dim=-1) - (probabilities * logits).sum(dim=-1)
    flat = logits.reshape(-1, logits.shape[-1])
    entropy, _ = _statistics(flat)
    return entropy.view(logits.shape[:-1])

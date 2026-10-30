"""Team-written tiled vocabulary reductions; no materialized log-softmax tensor.

Equations/semantics: torch.nn.functional.cross_entropy, Transformers 5.16.1
loss_utils.fixed_cross_entropy and TRL 0.29.1 trainer.utils.entropy_from_logits.
The Triton implementation here is original, not a copied kernel implementation.
"""

import torch
import triton
import triton.language as tl

TILE = 4096


@triton.jit
def vocab_tile_stats(x, partial_max, partial_sum, partial_weighted,
                     V: tl.constexpr, STRIDE: tl.constexpr, PARTS: tl.constexpr,
                     ENTROPY: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + row * STRIDE + col, col < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(value, 0)
    exponential = tl.exp(value - maximum)
    location = row * PARTS + part
    tl.store(partial_max + location, maximum)
    tl.store(partial_sum + location, tl.sum(exponential, 0))
    if ENTROPY:
        weighted = exponential * tl.where(col < V, value, 0.0)
        tl.store(partial_weighted + location, tl.sum(weighted, 0))


@triton.jit
def vocab_finish_stats(x, target, pm, ps, pw, lse, output,
                       V: tl.constexpr, STRIDE: tl.constexpr, PARTS: tl.constexpr,
                       IGNORE: tl.constexpr, ENTROPY: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    m = tl.load(pm + row * PARTS + part, part < PARTS, other=-float("inf"))
    s = tl.load(ps + row * PARTS + part, part < PARTS, other=0.0)
    largest = tl.max(m, 0)
    rescale = tl.exp(m - largest)
    total = tl.sum(rescale * s, 0)
    log_total = largest + tl.log(total)
    tl.store(lse + row, log_total)
    if ENTROPY:
        weighted = tl.load(pw + row * PARTS + part, part < PARTS, other=0.0)
        result = log_total - tl.sum(rescale * weighted, 0) / total
    else:
        label = tl.load(target + row)
        valid = label != IGNORE
        chosen = tl.load(x + row * STRIDE + label, valid, other=0.0).to(tl.float32)
        result = tl.where(valid, log_total - chosen, 0.0)
    tl.store(output + row, result)


@triton.jit
def vocab_ce_gradient(x, target, lse, upstream, dx,
                      V: tl.constexpr, STRIDE: tl.constexpr,
                      IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(target + row)
    value = tl.load(x + row * STRIDE + col, col < V, other=0.0).to(tl.float32)
    normalizer = tl.load(lse + row)
    scale = tl.load(upstream + row)
    gradient = (tl.exp(value - normalizer) - (col == label).to(tl.float32)) * scale
    gradient = tl.where(label != IGNORE, gradient, 0.0)
    tl.store(dx + row * V + col, gradient, col < V)


def _stats(x, target=None, ignore_index=-100):
    rows, vocab = x.shape
    parts = triton.cdiv(vocab, TILE)
    pm = torch.empty((rows, parts), device=x.device, dtype=torch.float32)
    ps = torch.empty_like(pm)
    entropy = target is None
    pw = torch.empty_like(pm) if entropy else ps
    lse = torch.empty(rows, device=x.device, dtype=torch.float32)
    out = torch.empty_like(lse)
    vocab_tile_stats[(rows, parts)](
        x, pm, ps, pw, vocab, x.stride(0), parts, entropy, TILE, num_warps=4,
    )
    vocab_finish_stats[(rows,)](
        x, target if target is not None else x, pm, ps, pw, lse, out,
        vocab, x.stride(0), parts, ignore_index, entropy,
        triton.next_power_of_2(parts), num_warps=4,
    )
    return out, lse


class _TokenCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, ignore_index):
        target = target.contiguous()
        loss, lse = _stats(x, target, ignore_index)
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad):
        x, target, lse = ctx.saved_tensors
        rows, vocab = x.shape
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        vocab_ce_gradient[(rows, triton.cdiv(vocab, TILE))](
            x, target, lse, grad.contiguous(), dx,
            vocab, x.stride(0), ctx.ignore_index, TILE, num_warps=4,
        )
        return dx, None, None


def token_cross_entropy(x, target, ignore_index=-100, reduction="mean"):
    """CUDA fp16/fp32 logits, fp32 reduction, integer labels, no smoothing."""
    losses = _TokenCrossEntropy.apply(x, target, ignore_index)
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    return total / (target != ignore_index).sum()


def token_entropy(logits):
    """Same leading dimensions as TRL's entropy function; inference-only metric."""
    shape = logits.shape[:-1]
    x = logits.reshape(-1, logits.shape[-1])
    out, _ = _stats(x)
    return out.reshape(shape)


def patch_vocabulary_reductions():
    import torch.nn.functional as functional
    from trl.trainer import sft_trainer

    if not getattr(functional.cross_entropy, "_soup_team_vocab", False):
        original = functional.cross_entropy

        def cross_entropy(input, target, weight=None, size_average=None,
                          ignore_index=-100, reduce=None, reduction="mean",
                          label_smoothing=0.0):
            supported = (
                isinstance(input, torch.Tensor) and input.is_cuda
                and input.dtype in (torch.float16, torch.float32) and input.ndim == 2
                and input.shape[0] > 0 and input.shape[1] >= 1024
                and input.stride(1) == 1 and target.is_cuda
                and target.device == input.device and target.ndim == 1
                and target.shape[0] == input.shape[0] and target.dtype == torch.long
                and weight is None and size_average is None and reduce is None
                and label_smoothing == 0.0 and reduction in ("none", "sum", "mean")
            )
            if supported:
                return token_cross_entropy(input, target, ignore_index, reduction)
            return original(input, target, weight, size_average, ignore_index,
                            reduce, reduction, label_smoothing)

        cross_entropy._soup_team_vocab = True
        functional.cross_entropy = cross_entropy

    if not getattr(sft_trainer.entropy_from_logits, "_soup_team_vocab", False):
        original_entropy = sft_trainer.entropy_from_logits

        def entropy_from_logits(logits, chunk_size=128):
            if (logits.is_cuda and logits.dtype in (torch.float16, torch.float32)
                    and logits.shape[-1] >= 1024 and logits.is_contiguous()
                    and not torch.is_grad_enabled()):
                return token_entropy(logits)
            return original_entropy(logits, chunk_size)

        entropy_from_logits._soup_team_vocab = True
        sft_trainer.entropy_from_logits = entropy_from_logits

"""Experimental FP32-math metrics from private FP16 logits.

No context management or output conversion lives here. The outer integration
must supply its strict, step-scoped predicate to install(). Outside that scope
all existing aliases are called unchanged. CPU imports do not require Triton.
"""

import functools

import torch


SUPPORTS_HALF_ENTROPY = True
SUPPORTS_HALF_REWIND = True


def supports_half_entropy(logits):
    return (isinstance(logits, torch.Tensor) and logits.is_cuda
            and logits.dtype == torch.float16 and logits.ndim >= 2
            and logits.numel() > 0 and logits.is_contiguous()
            and logits.shape[-1] > 0)


def supports_half_rows(logits, labels):
    return (supports_half_entropy(logits) and logits.ndim == 3
            and isinstance(labels, torch.Tensor)
            and labels.device == logits.device and labels.dtype == torch.int64
            and labels.shape == logits.shape[:2]
            and 2 <= logits.shape[1] <= 2**24)


@torch.no_grad()
def entropy_half(logits):
    """No-grad entropy of float32-promoted half values, without a dense upcast."""
    if not supports_half_entropy(logits):
        raise ValueError("unsupported private half logits for entropy")
    from .vocab import _statistics

    result, _unused_norm = _statistics(logits.view(-1, logits.shape[-1]))
    return result.view(logits.shape[:-1])


@torch.no_grad()
def row_losses_half(logits, labels):
    """Batch Soup's row CE without a dense logits slice/copy or scalar sync loop.

    Flat CE includes a final ignored target for every row, then its last column
    is excluded from row reduction, retaining Soup's causal alignment exactly.
    One small packed transfer returns means and counts; counts <=2**24 are
    exactly representable in float32 before conversion back to Python int.
    """
    if not supports_half_rows(logits, labels):
        raise ValueError("unsupported private half logits/labels for rewind")
    from .vocab import _half_cross_entropy

    batch, sequence, vocabulary = logits.shape
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)
    targets = targets[:, 1:].contiguous().view(-1)
    losses = _half_cross_entropy(logits.view(-1, vocabulary), targets,
                                 ignore_index=-100, reduction="none")
    token_losses = losses.view(batch, sequence)[:, :-1]
    mask = (labels[:, 1:] != -100).to(torch.float32)
    counts = mask.sum(dim=-1)
    totals = (token_losses * mask).sum(dim=-1)
    means = totals / counts.clamp(min=1.0)
    packed = torch.stack((means, counts), dim=-1).cpu().tolist()
    return [float(row[0]) for row in packed], [int(row[1]) for row in packed]


def install(scope_active):
    """Patch metric lookups only inside the caller's private training scope."""
    if not callable(scope_active):
        raise TypeError("scope_active must be callable")
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf

    original_entropy = sft_trainer.entropy_from_logits
    if not getattr(original_entropy, "_soup_private_half_metrics", False):
        @functools.wraps(original_entropy)
        def entropy(logits, *args, **kwargs):
            if (not args and not kwargs and scope_active()
                    and supports_half_entropy(logits)
                    and not (torch.is_grad_enabled() and logits.requires_grad)):
                return entropy_half(logits)
            return original_entropy(logits, *args, **kwargs)

        entropy._soup_private_half_metrics = True
        sft_trainer.entropy_from_logits = entropy

    original_rows = rewind_hf.row_losses
    if not getattr(original_rows, "_soup_private_half_metrics", False):
        @functools.wraps(original_rows)
        def rows(logits, labels):
            if scope_active() and supports_half_rows(logits, labels):
                return row_losses_half(logits, labels)
            return original_rows(logits, labels)

        rows._soup_private_half_metrics = True
        rewind_hf.row_losses = rows

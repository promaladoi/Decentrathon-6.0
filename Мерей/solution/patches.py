"""Narrow Soup/Transformers/TRL integration; eager fallbacks stay available."""

import functools

import torch


def _can_reduce(logits):
    return (isinstance(logits, torch.Tensor) and logits.is_cuda
            and logits.dtype == torch.float32 and logits.ndim >= 2
            and logits.numel() > 0 and logits.is_contiguous()
            and logits.shape[-1] > 0)


def _can_ce(logits, targets):
    return (_can_reduce(logits) and logits.ndim == 2
            and isinstance(targets, torch.Tensor) and targets.device == logits.device
            and targets.dtype == torch.int64 and targets.ndim == 1
            and targets.shape[0] == logits.shape[0] and targets.is_contiguous())


def install_loss_patch():
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    if getattr(original, "_soup_vocab_patch", False):
        return

    @functools.wraps(original)
    def fixed(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
        if not _can_ce(source, target):
            return original(source, target, num_items_in_batch, ignore_index, **kwargs)
        from .vocab import cross_entropy

        loss = cross_entropy(source, target, ignore_index,
                             "sum" if num_items_in_batch is not None else "mean")
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss

    fixed._soup_vocab_patch = True
    loss_utils.fixed_cross_entropy = fixed


def install_entropy_patch():
    # Patch the alias actually looked up by SFTTrainer.compute_loss; keep the
    # trainer's original masks, accuracy and metric aggregation.
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    if getattr(original, "_soup_vocab_patch", False):
        return

    @functools.wraps(original)
    def entropy(logits, *args, **kwargs):
        if (args or kwargs or not _can_reduce(logits)
                or (torch.is_grad_enabled() and logits.requires_grad)):
            return original(logits, *args, **kwargs)
        from .vocab import entropy_from_logits

        return entropy_from_logits(logits)

    entropy._soup_vocab_patch = True
    sft_trainer.entropy_from_logits = entropy


def install_rewind_patch():
    from soup_cli.trainer import rewind_hf

    original = rewind_hf.row_losses
    if getattr(original, "_soup_vocab_patch", False):
        return

    @functools.wraps(original)
    def row_losses(logits, labels):
        if (not _can_reduce(logits) or logits.ndim != 3
                or not isinstance(labels, torch.Tensor)
                or labels.dtype != torch.int64 or labels.device != logits.device
                or labels.shape != logits.shape[:2] or logits.shape[1] < 2):
            return original(logits, labels)
        from .vocab import cross_entropy

        values, counts = [], []
        with torch.no_grad():
            # Keep Soup's per-row accumulation and causal alignment, including
            # the 0.0 loss for a row without supervised tokens. No logits copy.
            for row in range(labels.shape[0]):
                target = labels[row, 1:].contiguous()
                token_loss = cross_entropy(logits[row, :-1].detach(), target,
                                           reduction="none")
                mask = (target != -100).to(token_loss.dtype)
                count = mask.sum()
                values.append(float((token_loss * mask).sum() / count.clamp(min=1.0)))
                counts.append(int(count.item()))
        return values, counts

    row_losses._soup_vocab_patch = True
    rewind_hf.row_losses = row_losses


def install_model_patches():
    install_loss_patch()
    install_entropy_patch()
    install_rewind_patch()
    if torch.cuda.is_available():
        from .rmsnorm import patch_qwen2_rmsnorm
        from .rope import patch_qwen2_rope
        from .nf4 import patch_nf4_dequant
        from .swiglu import patch_qwen2_mlp
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        patch_qwen2_rmsnorm()
        patch_qwen2_rope()
        patch_nf4_dequant()
        # The starter example is allowed alongside original kernels. Keep its
        # memory optimization, with explicit attribution and no claim to its
        # time in the >=5% requirement. Its own patch is not idempotent.
        if not getattr(Qwen2MLP.forward, "_soup_example_installed", False):
            mlp_original = Qwen2MLP.forward
            patch_qwen2_mlp()
            Qwen2MLP.forward.__wrapped__ = mlp_original
            Qwen2MLP.forward._soup_example_installed = True


def patch_soup_setup():
    from soup_cli.trainer.sft import SFTTrainerWrapper

    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_soup_solution_patch", False):
        return

    @functools.wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if isinstance(getattr(self, "model", None), torch.nn.Module):
            install_model_patches()
        return result

    setup._soup_solution_patch = True
    SFTTrainerWrapper._setup_transformers = setup

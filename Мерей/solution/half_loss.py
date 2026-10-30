"""Experimental instance-only Qwen2 half-input causal loss.

This module imports no Triton on CPU. It preserves the model's original
forward/output path, including Accelerate's normal float32 output conversion.
Install only after the real PEFT model exists; no global loss mapping changes.
"""

import functools

import torch


def _supported(logits, labels, vocab_size, shift_labels):
    return (isinstance(logits, torch.Tensor) and logits.is_cuda
            and logits.dtype == torch.float16 and logits.ndim == 3
            and logits.numel() > 0 and logits.is_contiguous()
            and logits.shape[-1] == vocab_size
            and isinstance(labels, torch.Tensor) and labels.device == logits.device
            and labels.dtype == torch.int64 and labels.shape == logits.shape[:-1]
            and shift_labels is None)


def _wrap_loss(original):
    @functools.wraps(original)
    def loss(logits, labels, vocab_size, num_items_in_batch=None,
             ignore_index=-100, shift_labels=None, **kwargs):
        if not _supported(logits, labels, vocab_size, shift_labels):
            return original(logits, labels, vocab_size, num_items_in_batch,
                            ignore_index, shift_labels, **kwargs)
        # Lazy import: CPU test setup and unsupported cases need no Triton.
        from .vocab import _half_cross_entropy

        shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shifted = shifted[..., 1:].contiguous().view(-1)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        result = _half_cross_entropy(logits.view(-1, vocab_size), shifted,
                                     ignore_index, reduction)
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(result.device)
            result = result / num_items_in_batch
        return result

    loss._soup_half_loss_patch = True
    return loss


def install(model):
    """Return installed instance count; preserve custom losses and CPU models."""
    if (not isinstance(model, torch.nn.Module) or not torch.cuda.is_available()
            or not any(parameter.is_cuda for parameter in model.parameters())):
        return 0
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    installed = 0
    for layer in model.modules():
        if not isinstance(layer, Qwen2ForCausalLM):
            continue
        original = layer.loss_function
        if original is not ForCausalLMLoss:
            continue
        layer.loss_function = _wrap_loss(original)
        installed += 1
    return installed

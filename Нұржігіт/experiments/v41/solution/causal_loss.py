"""Causal-LM loss semantics from Transformers 5.16.1 loss_utils (Apache-2.0).

Read half logits directly; the existing team Triton reductions calculate in
float32 and return float32 loss, and gradients are stored in input dtype.
"""
import torch
from .vocab import token_cross_entropy


def causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        shift_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    targets = shift_labels.reshape(-1).to(logits.device)
    reduction = 'sum' if num_items_in_batch is not None else 'mean'
    loss = token_cross_entropy(logits.view(-1, vocab_size), targets, ignore_index, reduction)
    if num_items_in_batch is not None:
        count = num_items_in_batch.to(loss.device) if torch.is_tensor(num_items_in_batch) else num_items_in_batch
        loss = loss / count
    return loss


def patch_causal_loss():
    from transformers.loss import loss_utils
    original = loss_utils.LOSS_MAPPING['ForCausalLM']
    if getattr(original, '_team_causal_loss', False):
        return

    def forward(logits, labels, vocab_size, num_items_in_batch=None,
                ignore_index=-100, shift_labels=None, **kwargs):
        targets = labels if shift_labels is None else shift_labels
        if (logits.is_cuda and logits.is_contiguous() and logits.numel() > 0
                and logits.dtype in (torch.float16, torch.float32)
                and logits.shape[-1] == vocab_size and vocab_size >= 1024
                and targets is not None and targets.dtype == torch.long):
            return causal_loss(logits, labels, vocab_size, num_items_in_batch,
                               ignore_index, shift_labels, **kwargs)
        return original(logits, labels, vocab_size, num_items_in_batch,
                        ignore_index, shift_labels, **kwargs)

    forward._team_causal_loss = True
    loss_utils.LOSS_MAPPING['ForCausalLM'] = forward

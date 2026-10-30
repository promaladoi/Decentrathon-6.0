"""Lossless storage for a frozen, tied embedding/head pair after k-bit preparation.

PEFT upcasts these weights to fp32. If every value round-trips exactly through
fp16, store that same Parameter in fp16 and retain the original fp32 embedding
output and non-autocast linear contract. No trainable parameter is compressed.
The all-or-nothing value check runs once, before the first model forward.
"""
from functools import wraps
from types import MethodType

import torch
from torch import nn


def _unobserved(module):
    return not (module._forward_hooks or module._forward_pre_hooks or
                module._backward_hooks or getattr(module, '_backward_pre_hooks', None) or
                hasattr(module, '_hf_hook'))


def _eligible(embedding, head):
    if type(embedding) is not nn.Embedding or type(head) is not nn.Linear:
        return False
    weight = embedding.weight
    if (head.weight is not weight or type(weight) is not nn.Parameter or
            weight.requires_grad or weight.grad is not None or weight.dtype != torch.float32 or
            not weight.is_contiguous() or weight.ndim != 2 or weight.shape[0] < 1024 or
            embedding.max_norm is not None or head.bias is not None or
            head.in_features != weight.shape[1] or head.out_features != weight.shape[0]):
        return False
    if not _unobserved(embedding) or not _unobserved(head):
        return False
    if (getattr(embedding.forward, '__func__', None) is not nn.Embedding.forward or
            getattr(head.forward, '__func__', None) is not nn.Linear.forward):
        return False
    # A global hook could observe the internal storage dtype. Keep its ordinary path.
    hooks = torch.nn.modules.module
    return not any(getattr(hooks, name, None) for name in
                   ('_global_forward_hooks', '_global_forward_pre_hooks',
                    '_global_backward_hooks', '_global_backward_pre_hooks'))


def exact_half_copy(weight):
    """Return a compact copy only if every finite fp32 value is preserved exactly."""
    if weight.dtype != torch.float32 or not weight.is_contiguous():
        return None
    if weight.is_cuda:
        from .frozen_storage_kernel import exact_half_copy_cuda
        return exact_half_copy_cuda(weight)
    packed = weight.detach().to(torch.float16)
    if torch.isfinite(weight).all().item() and torch.equal(packed.float(), weight):
        return packed
    return None


def _restore_for_training(module):
    state = module.__dict__['_solution_frozen_storage']
    weight = state['weight']
    if weight.requires_grad and weight.dtype == torch.float16:
        # The optimization is exclusively for frozen weights. Restore the
        # original storage before any later caller chooses to train this pair.
        with torch.no_grad():
            weight.data = weight.data.float()
        state['active'] = False
    return state


def _embedding_forward(self, input):
    state = _restore_for_training(self)
    result = nn.Embedding.forward(self, input)
    if self.weight is state['weight'] and self.weight.dtype == torch.float16:
        return result.float()
    return result


def _head_forward(self, input):
    state = _restore_for_training(self)
    if self.weight is not state['weight'] or self.weight.dtype != torch.float16:
        return nn.Linear.forward(self, input)
    if torch.is_autocast_enabled(input.device.type):
        return nn.Linear.forward(self, input)
    # Outside AMP the old fp32 weight must still provide fp32 accumulation and
    # input gradients. This temporary exists only on the fallback/eval path.
    return torch.nn.functional.linear(input, self.weight.float(), self.bias)


def install_frozen_storage(embedding, head):
    """Install once on an eligible tied pair; return diagnostic scalar metadata."""
    if '_solution_frozen_storage' in embedding.__dict__:
        return embedding.__dict__['_solution_frozen_storage']
    if not _eligible(embedding, head):
        return None
    weight = embedding.weight
    with torch.no_grad():
        packed = exact_half_copy(weight)
    if packed is None:
        return None
    state = {'weight': weight, 'active': True, 'elements': weight.numel(),
             'storage_bytes_saved': weight.numel() * 2}
    # No reference to the previous fp32 storage is kept. The Parameter itself,
    # tied identity, trainable set and state_dict keys remain the same.
    weight.data = packed
    embedding.__dict__['_solution_frozen_storage'] = state
    head.__dict__['_solution_frozen_storage'] = state
    embedding.forward = MethodType(_embedding_forward, embedding)
    head.forward = MethodType(_head_forward, head)
    return state


def prepare_trainer_storage(trainer):
    """Pack an already-prepared native Qwen pair before later setup allocations.

    Called only after the native private trainer was successfully wrapped.
    The existing value proof and public output wrappers are unchanged.
    """
    from peft import PeftModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    model = getattr(trainer, 'model', None)
    if type(model) is PeftModelForCausalLM:
        model = model.get_base_model()
    if type(model) is not Qwen2ForCausalLM:
        return None
    embedding, head = model.model.embed_tokens, model.lm_head
    if not embedding.weight.is_cuda:
        return None
    state = install_frozen_storage(embedding, head)
    if state is not None:
        model.__dict__['_solution_storage_attempted'] = True
        print('solution: early lossless frozen tied storage saved '
              f"{state['storage_bytes_saved']} bytes", flush=True)
    return state


def patch_frozen_storage():
    """Wrap the final Qwen forward after the team's unpadding patch is installed."""
    from transformers.models.qwen2 import modeling_qwen2
    cls = modeling_qwen2.Qwen2ForCausalLM
    original = cls.forward
    if getattr(original, '_solution_frozen_storage', False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        if not self.__dict__.get('_solution_storage_attempted', False):
            self.__dict__['_solution_storage_attempted'] = True
            embedding = getattr(getattr(self, 'model', None), 'embed_tokens', None)
            head = getattr(self, 'lm_head', None)
            if (type(embedding) is nn.Embedding and type(head) is nn.Linear and
                    embedding.weight.is_cuda):
                state = install_frozen_storage(embedding, head)
                if state is not None:
                    print('solution: lossless frozen tied storage saved '
                          f"{state['storage_bytes_saved']} bytes", flush=True)
        return original(self, *args, **kwargs)

    forward._solution_frozen_storage = True
    cls.forward = forward

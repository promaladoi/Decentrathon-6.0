"""Experimental frozen LM-head autograd: recompute its temporary half weight.

Uses the same native linear/mm operations and dtype conversions. A frozen fp32
parameter is retained; its fp16 autocast copy is not held across the forward.
Only plain, unobserved Qwen2 LM heads under native CUDA fp16 AMP are eligible.
"""
import functools
from types import MethodType
import torch
import torch.nn.functional as F

_NATIVE_LINEAR_FORWARD = torch.nn.Linear.forward
_NATIVE_FUNCTIONAL_LINEAR = F.linear


def _canonical_linear():
    """Reject a replaced class or functional implementation of native linear."""
    return (torch.nn.Linear.forward is _NATIVE_LINEAR_FORWARD
            and F.linear is _NATIVE_FUNCTIONAL_LINEAR
            and getattr(_NATIVE_LINEAR_FORWARD, '__module__', None) == 'torch.nn.modules.linear'
            and getattr(_NATIVE_LINEAR_FORWARD, '__qualname__', None) == 'Linear.forward'
            and not hasattr(_NATIVE_LINEAR_FORWARD, '__wrapped__'))


def _unobserved(layer):
    keys = ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')
    global_keys = ('_global_forward_hooks', '_global_forward_pre_hooks',
                   '_global_backward_hooks', '_global_backward_pre_hooks')
    return (not any(getattr(torch.nn.modules.module, key, None) for key in global_keys)
            and not any(getattr(layer, key, None) for key in keys))


class _FrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        output = F.linear(x, weight)
        ctx.save_for_backward(weight)
        ctx.input_shape, ctx.input_dtype = x.shape, x.dtype
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        compute_weight = weight.to(ctx.compute_dtype)
        gradient = torch.mm(grad_output.reshape(-1, weight.shape[0]).to(ctx.compute_dtype),
                            compute_weight)
        return gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None


def _supported(layer, x):
    if (type(layer) is not torch.nn.Linear or layer.bias is not None
            or layer.weight.requires_grad or layer.weight.dtype != torch.float32
            or not layer.weight.is_cuda or not layer.weight.is_contiguous()
            or not isinstance(x, torch.Tensor) or not x.is_cuda
            or x.device != layer.weight.device or x.dtype != torch.float32
            or x.ndim < 2 or x.numel() == 0 or not x.is_contiguous()
            or x.shape[-1] != layer.weight.shape[1] or not x.requires_grad
            or not torch.is_grad_enabled() or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.float16):
        return False
    if not _canonical_linear() or not _unobserved(layer):
        return False
    bound = getattr(layer.forward, '__func__', None)
    return (bound is _NATIVE_LINEAR_FORWARD
            or getattr(bound, '_soup_frozen_head_original', None) is _NATIVE_LINEAR_FORWARD)


def install(model):
    if (not isinstance(model, torch.nn.Module) or not torch.cuda.is_available()
            or not any(parameter.is_cuda for parameter in model.parameters())
            or not _canonical_linear()):
        return 0
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    count = 0
    for parent in model.modules():
        if type(parent) is not Qwen2ForCausalLM:
            continue
        head = parent.lm_head
        original = getattr(head, 'forward', None)
        if (type(head) is not torch.nn.Linear
                or getattr(original, '_soup_frozen_head', False)
                or getattr(original, '__func__', None) is not _NATIVE_LINEAR_FORWARD
                or head.bias is not None or head.weight.requires_grad
                or head.weight.dtype != torch.float32 or not head.weight.is_cuda
                or not head.weight.is_contiguous() or not _unobserved(head)):
            continue
        @functools.wraps(original)
        def forward(self, x, _original=original):
            if not _supported(self, x):
                return _original(x)
            return _FrozenHead.apply(x, self.weight)
        forward._soup_frozen_head = True
        forward._soup_frozen_head_original = _NATIVE_LINEAR_FORWARD
        head.forward = MethodType(forward, head)
        count += 1
    return count

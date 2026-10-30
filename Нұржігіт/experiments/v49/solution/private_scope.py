"""Keep teammate streaming logits strictly inside Soup's private loss consumer.

Public model calls and outer compute_loss(return_outputs=True) take the ordinary
Tensor-logits forward. The ContextVar is reset on every exit, and binds one
exact Qwen instance and the exact tensors passed to the outer recorder.
"""
import contextvars
import functools
import math
from dataclasses import dataclass
from types import MethodType
import torch


@dataclass(frozen=True)
class _Scope:
    base: object
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    loss_scale: object = None


_SCOPE = contextvars.ContextVar('soup_private_streaming_logits', default=None)


def expected_loss_scale():
    scope = _SCOPE.get()
    return None if scope is None else scope.loss_scale


def permits(model, input_ids, labels, attention_mask):
    scope = _SCOPE.get()
    return (scope is not None and scope.base is model
            and scope.input_ids is input_ids and scope.labels is labels
            and scope.attention_mask is attention_mask)


def _observers(model):
    from torch.nn.modules import module
    if any(getattr(module, name, None) for name in
           ('_global_forward_hooks', '_global_forward_pre_hooks',
            '_global_backward_hooks', '_global_backward_pre_hooks')):
        return True
    return any(getattr(model, name, None) for name in
               ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))


def _scope_for_call(trainer, model, inputs, return_outputs):
    if (return_outputs is not False or _SCOPE.get() is not None
            or not torch.cuda.is_available() or not torch.is_grad_enabled()
            or not getattr(model, 'training', False) or not isinstance(inputs, dict)
            or _observers(model)):
        return None
    state = getattr(trainer, '_soup_rewind_state', None)
    args = getattr(trainer, 'args', None)
    accelerator = getattr(trainer, 'accelerator', None)
    if (not getattr(trainer, 'soup_rewind', False) or state is None
            or getattr(state, 'failed', True) or args is None or accelerator is None
            or getattr(trainer, 'compute_loss_func', None) is not None
            or getattr(trainer, 'label_smoother', None) is not None
            or getattr(trainer, 'aux_loss_enabled', False)
            or getattr(trainer, 'num_virtual_tokens', 0) != 0
            or getattr(trainer, 'padding_free', False)
            or not getattr(args, 'fp16', False)
            or getattr(args, 'loss_type', 'nll') != 'nll'
            or getattr(args, 'past_index', -1) >= 0
            or any(getattr(args, key, False) for key in
                   ('use_liger_kernel', 'packing', 'gradient_checkpointing',
                    'activation_offloading', 'torch_compile'))
            or getattr(accelerator, 'num_processes', 1) != 1
            or getattr(accelerator, 'parallelism_config', None) is not None
            or not getattr(accelerator, 'native_amp', False)
            or getattr(accelerator, 'mixed_precision', None) != 'fp16'):
        return None
    if (inputs.get('position_ids') is not None
            or inputs.get('past_key_values') is not None
            or inputs.get('inputs_embeds') is not None
            or 'shift_labels' in inputs
            or type(inputs.get('logits_to_keep', 0)) is not int
            or inputs.get('logits_to_keep', 0) != 0
            or inputs.get('return_dict') is False
            or inputs.get('output_attentions')
            or inputs.get('output_hidden_states')):
        return None
    ids, labels, mask = (inputs.get(key) for key in
                         ('input_ids', 'labels', 'attention_mask'))
    if (not all(isinstance(x, torch.Tensor) and x.is_cuda for x in (ids, labels, mask))
            or ids.dtype != torch.int64 or labels.dtype != torch.int64
            or ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 2
            or labels.shape != ids.shape or mask.shape != ids.shape
            or not (ids.device == labels.device == mask.device)):
        return None
    from peft import PeftModelForCausalLM, PeftType
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    base = accelerator.unwrap_model(model)
    if type(base) is PeftModelForCausalLM:
        config = base.active_peft_config
        if config.is_prompt_learning or config.peft_type != PeftType.LORA:
            return None
        base = base.get_base_model()
    if (type(base) is not Qwen2ForCausalLM
            or not getattr(base.forward, '_soup_private_unpad', False)):
        return None
    expected_scale = None
    scaler = getattr(accelerator, 'scaler', None)
    from torch.amp.grad_scaler import GradScaler
    if (type(scaler) is GradScaler and scaler.is_enabled()
            and getattr(accelerator, 'gradient_accumulation_steps', 1) == 1
            and getattr(args, 'gradient_accumulation_steps', 1) == 1):
        # Snapshot the native scaler for this call, including its initial
        # pre-initialization value. A changed upstream gets a real fallback
        # in lmhead.backward; no gradient is rescaled after half GEMM.
        current_scale = float(scaler.get_scale())
        if math.isfinite(current_scale) and current_scale > 0:
            expected_scale = current_scale
    return _Scope(base, ids, labels, mask, expected_scale)


def wrap_trainer(trainer):
    original = getattr(trainer, 'compute_loss', None)
    function = getattr(original, '__func__', None)
    if (not callable(original) or getattr(original, '_soup_private_streaming', False)
            or function is None or function.__module__ != 'soup_cli.trainer.rewind_hf'
            or not function.__qualname__.endswith('RewindTrainer.compute_loss')):
        return False

    @functools.wraps(function)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scope = _scope_for_call(self, model, inputs, return_outputs)
        token = _SCOPE.set(scope)
        try:
            return original(model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        finally:
            _SCOPE.reset(token)

    compute_loss._soup_private_streaming = True
    trainer.compute_loss = MethodType(compute_loss, trainer)
    return True


def install():
    from soup_cli.trainer.sft import SFTTrainerWrapper
    original = SFTTrainerWrapper._build_rewind_trainer
    if getattr(original, '_soup_private_streaming', False):
        return False

    @functools.wraps(original)
    def build(self, *args, **kwargs):
        trainer = original(self, *args, **kwargs)
        if wrap_trainer(trainer):
            from .frozen_storage import prepare_trainer_storage
            prepare_trainer_storage(trainer)
        return trainer

    build._soup_private_streaming = True
    SFTTrainerWrapper._build_rewind_trainer = build
    return True

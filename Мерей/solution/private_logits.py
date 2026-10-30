"""Experimental private logits lifetime optimization for the fixed Soup trainer.

Only the outermost Soup Rewind compute_loss call may temporarily retain the
model's fp16 logits while its internal metrics run. Public model calls keep
Accelerate's original output conversion. A compute_loss caller that requests
outputs receives the original fp32 conversion before this wrapper returns.

This does not change training settings, model forward, loss, recorder order,
or TRL's metric formulas. The separate half_metrics module performs internal
entropy/row-loss arithmetic in fp32. No tensor is cached between calls.
"""

import contextvars
import functools
from dataclasses import dataclass
from types import MethodType

import torch


@dataclass(frozen=True)
class _Scope:
    labels: torch.Tensor
    shape: tuple


_PRIVATE_SCOPE = contextvars.ContextVar("soup_private_logits_scope", default=None)
_ORIGINAL_CONVERT = None
_HALF_METRICS = None


def private_logits_active():
    """Predicate supplied to the half-metric aliases; scoped to this call."""
    return _PRIVATE_SCOPE.get() is not None


def _can_keep_logits(outputs, scope):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if (scope is None or _HALF_METRICS is None
            or type(outputs) is not CausalLMOutputWithPast
            or set(outputs.keys()) != {"loss", "logits"}):
        return False
    logits = outputs.logits
    return (isinstance(logits, torch.Tensor) and logits.is_cuda
            and logits.dtype == torch.float16 and logits.requires_grad
            and tuple(logits.shape) == scope.shape
            and logits.device == scope.labels.device and logits.is_contiguous()
            and _HALF_METRICS.supports_half_entropy(logits)
            and _HALF_METRICS.supports_half_rows(logits, scope.labels))


def _install_output_conversion():
    global _ORIGINAL_CONVERT
    from accelerate.utils import operations

    current = operations.convert_to_fp32
    if getattr(current, "_soup_private_logits_owner", None) is _PRIVATE_SCOPE:
        # Recover the exact converter captured by the already-installed
        # closure, even if a test or module-state restore cleared our reference.
        # Idempotent installation must leave the public restoration path valid.
        _ORIGINAL_CONVERT = getattr(current, "_soup_private_logits_original",
                                    getattr(current, "__wrapped__", None))
        if not callable(_ORIGINAL_CONVERT):
            raise RuntimeError("installed private converter lost its original callable")
        return False
    _ORIGINAL_CONVERT = current

    @functools.wraps(current)
    def convert(outputs):
        scope = _PRIVATE_SCOPE.get()
        if scope is None or not _can_keep_logits(outputs, scope):
            return current(outputs)
        # Accelerate's original recursively_apply reconstructs this same
        # ModelOutput type. Convert every other field through that same path.
        return type(outputs)({key: value if key == "logits" else current(value)
                              for key, value in outputs.items()})

    convert._soup_private_logits_owner = _PRIVATE_SCOPE
    convert._soup_private_logits_original = current
    operations.convert_to_fp32 = convert
    return True


def _has_output_observer(model):
    # A forward hook on the prepared outer model can observe its AMP-converted
    # output before compute_loss returns. Such calls keep the original path.
    from torch.nn.modules import module

    if (getattr(module, "_global_forward_hooks", None)
            or getattr(module, "_global_forward_pre_hooks", None)):
        return True
    return any(getattr(model, name, None)
               for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))


def _scope_for_call(trainer, model, inputs, return_outputs):
    if (private_logits_active() or _HALF_METRICS is None
            or not torch.cuda.is_available() or not torch.is_grad_enabled()
            or not getattr(model, "training", False)
            or not isinstance(return_outputs, bool) or not isinstance(inputs, dict)
            or "shift_labels" in inputs or _has_output_observer(model)):
        return None
    state = getattr(trainer, "_soup_rewind_state", None)
    args = getattr(trainer, "args", None)
    accelerator = getattr(trainer, "accelerator", None)
    if (not getattr(trainer, "soup_rewind", False) or state is None
            or getattr(state, "failed", True)
            or getattr(trainer, "compute_loss_func", None) is not None
            or getattr(trainer, "label_smoother", None) is not None
            or getattr(trainer, "aux_loss_enabled", False)
            or getattr(trainer, "num_virtual_tokens", 0) != 0
            or getattr(trainer, "padding_free", False)
            or args is None or accelerator is None
            or getattr(args, "use_liger_kernel", False)
            or getattr(args, "packing", False)
            or getattr(args, "gradient_checkpointing", False)
            or getattr(args, "activation_offloading", False)
            or getattr(args, "torch_compile", False)
            or getattr(args, "past_index", -1) >= 0
            or getattr(args, "loss_type", "nll") != "nll"
            or not getattr(args, "fp16", False)
            or getattr(accelerator, "num_processes", 1) != 1
            or getattr(accelerator, "parallelism_config", None) is not None
            or not getattr(accelerator, "native_amp", False)
            or getattr(accelerator, "mixed_precision", None) != "fp16"):
        return None
    if (type(inputs.get("logits_to_keep", 0)) is not int
            or inputs.get("logits_to_keep", 0) != 0
            or inputs.get("return_dict", None) is False):
        return None
    for flag in ("output_hidden_states", "output_attentions"):
        if inputs.get(flag, None) is not None and inputs.get(flag) is not False:
            return None
    labels = inputs.get("labels")
    if (not isinstance(labels, torch.Tensor) or not labels.is_cuda
            or labels.dtype != torch.int64 or labels.ndim != 2
            or labels.shape[0] < 1 or labels.shape[1] < 2):
        return None

    from peft import PeftModelForCausalLM, PeftType
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    unwrapped = accelerator.unwrap_model(model)
    if type(unwrapped) is PeftModelForCausalLM:
        config = unwrapped.active_peft_config
        if config.is_prompt_learning or config.peft_type != PeftType.LORA:
            return None
        base = unwrapped.get_base_model()
    else:
        base = unwrapped
    if type(base) is not Qwen2ForCausalLM:
        return None
    # The installed v9 loss retains fp16 logits and does all CE math in fp32.
    # Without that narrow loss hook, no private half path is enabled.
    if not getattr(base.loss_function, "_soup_half_loss_patch", False):
        return None
    return _Scope(labels=labels,
                  shape=(*labels.shape, int(base.config.vocab_size)))


def wrap_trainer(trainer):
    """Wrap the actual outer Soup recorder instance, preserving its class."""
    original = getattr(trainer, "compute_loss", None)
    if (not callable(original) or not getattr(trainer, "soup_rewind", False)
            or getattr(original, "_soup_private_logits_patch", False)):
        return False
    # Do not wrap an unrelated/custom outer override: the recorder must be
    # the last internal consumer before any public return_outputs boundary.
    function = getattr(original, "__func__", None)
    if (function is None
            or function.__module__ != "soup_cli.trainer.rewind_hf"
            or not function.__qualname__.endswith("RewindTrainer.compute_loss")):
        return False

    @functools.wraps(function)
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        scope = _scope_for_call(self, model, inputs, return_outputs)
        # Set None as well: a nested unsupported call must suspend any outer
        # private context, otherwise it could expose half outputs to its caller.
        token = _PRIVATE_SCOPE.set(scope)
        try:
            result = original(model, inputs, return_outputs=return_outputs,
                              num_items_in_batch=num_items_in_batch)
        finally:
            _PRIVATE_SCOPE.reset(token)
        if scope is not None and (return_outputs or not isinstance(result, torch.Tensor)):
            # The boundary is outside the context and uses Accelerate's exact
            # original converter, preserving returned type, fields and dtypes.
            return _ORIGINAL_CONVERT(result)
        return result

    compute_loss._soup_private_logits_patch = True
    trainer.compute_loss = MethodType(compute_loss, trainer)
    return True


def install():
    """Install narrow experiment hooks; production callers still need tests."""
    global _HALF_METRICS
    from . import half_metrics
    from soup_cli.trainer.sft import SFTTrainerWrapper

    half_metrics.install(private_logits_active)
    _HALF_METRICS = half_metrics
    _install_output_conversion()
    original = SFTTrainerWrapper._build_rewind_trainer
    if getattr(original, "_soup_private_logits_patch", False):
        return False

    @functools.wraps(original)
    def build(self, *args, **kwargs):
        trainer = original(self, *args, **kwargs)
        wrap_trainer(trainer)
        return trainer

    build._soup_private_logits_patch = True
    SFTTrainerWrapper._build_rewind_trainer = build
    return True

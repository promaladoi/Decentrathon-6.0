"""Experimental own frozen-weight offload; no Soup setting is changed.

The native Transformers loader uses CPU placement. Each ordinary NF4
quantization still runs through native bitsandbytes on the execution GPU,
then its exact bytes and scales move to CPU. LoRA initialization, preparation
and all task arguments remain native; only frozen-weight placement changes.
This module is a candidate until full external-peak and admission checks pass.
"""
import copy
import functools
import threading
from types import MethodType

import torch
from torch import nn


_LOAD_LOCK = threading.RLock()


def _pin_state(state):
    for name in ('absmax', 'code', 'offset'):
        tensor = getattr(state, name, None)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            setattr(state, name, tensor if tensor.is_pinned() else tensor.pin_memory())
    if getattr(state, 'state2', None) is not None:
        _pin_state(state.state2)
    return state


def streamed(packed, state):
    return (isinstance(packed, torch.Tensor) and packed.device.type == 'cpu'
            and bool(getattr(state, '_solution_cpu_streamed', False))
            and packed.dtype == torch.uint8 and not packed.requires_grad)


def transfer_nf4(packed, state, device=None):
    """A fresh bounded copy per use: no stale cache or mutation of CPU sources."""
    if not streamed(packed, state):
        raise ValueError('not an owned CPU-streamed NF4 parameter')
    device = torch.device(device if device is not None else state._solution_execution_device)
    # QuantState.to mutates its object, so give each operation independent
    # metadata. Only the small scales are copied; no dense weight is kept.
    # QuantState metadata is shallow copied while every tensor attribute gets
    # an independent GPU tensor. This avoids cloning the pinned CPU scales
    # before every transfer. Nested state is a distinct object as well.
    def transferred(source):
        result = copy.copy(source)
        for name in ('absmax', 'code', 'offset'):
            tensor = getattr(source, name, None)
            if isinstance(tensor, torch.Tensor):
                setattr(result, name, tensor.to(device, non_blocking=True))
        if getattr(source, 'state2', None) is not None:
            result.state2 = transferred(source.state2)
        return result
    live = transferred(state)
    dense_bytes = packed.detach().as_subclass(torch.Tensor).to(device, non_blocking=True)
    return dense_bytes, live


def _embedding_forward(self, indices):
    if self.weight.device.type != 'cpu':
        return nn.Embedding.forward(self, indices)
    cpu_indices = indices.detach().to('cpu')
    out = nn.Embedding.forward(self, cpu_indices)
    if self.__dict__.get('_solution_original_embedding_dtype') == torch.float32:
        out = out.float()
    if indices.is_cuda:
        out = out.pin_memory().to(indices.device, non_blocking=True)
    return out


def _head_forward(self, x):
    weight = self.weight
    if weight.device != x.device:
        dtype = torch.float16 if torch.is_autocast_enabled(x.device.type) else torch.float32
        weight = weight.to(device=x.device, dtype=dtype)
    return torch.nn.functional.linear(x, weight, self.bias)


def prepare_streamed_model(model):
    """Run after native PEFT creation, before trainer/optimizer preparation."""
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if not bool(base.__dict__.get('_solution_cpu_streamed', False)):
        return model
    device = base.__dict__['_solution_execution_device']
    embedding, head = base.model.embed_tokens, base.lm_head
    if embedding.weight is not head.weight:
        raise RuntimeError('CPU streaming requires the original tied embedding/head')
    from .frozen_storage import install_frozen_storage
    original_dtype = embedding.weight.dtype
    storage = install_frozen_storage(embedding, head)
    if storage is None or embedding.weight.dtype != torch.float16:
        raise RuntimeError('frozen tied weight did not pass exact-half storage proof')
    embedding.weight.data = embedding.weight.data.pin_memory()
    embedding.__dict__['_solution_original_embedding_dtype'] = original_dtype
    embedding.forward = MethodType(_embedding_forward, embedding)
    head.forward = MethodType(_head_forward, head)
    head.weight._solution_cpu_streamed = True
    kept = {id(embedding.weight), id(head.weight)}
    import bitsandbytes as bnb
    for parameter in model.parameters():
        if isinstance(parameter, bnb.nn.Params4bit):
            if not streamed(parameter, parameter.quant_state):
                raise RuntimeError('loader failed to offload an NF4 weight')
        elif id(parameter) not in kept:
            parameter.data = parameter.data.to(device)
    for module in model.modules():
        for key, buffer in tuple(module._buffers.items()):
            if isinstance(buffer, torch.Tensor) and not buffer.is_meta:
                module._buffers[key] = buffer.to(device)
    # This describes actual storage placement, not a fictitious all-GPU map.
    # bitsandbytes' multi-backend support permits mixed CPU/CUDA parameters;
    # this solution implements their transfers rather than Accelerate hooks.
    placement = {}
    for name, parameter in base.named_parameters():
        placement[name] = str(parameter.device)
    placement['lm_head.weight'] = 'cpu'
    base.hf_device_map = placement
    base.__dict__['_solution_cpu_streaming_ready'] = True
    print('solution: own CPU frozen-weight streaming enabled', flush=True)
    return model


def _eligible_load(name, kwargs):
    config = kwargs.get('quantization_config')
    return (name == 'Qwen/Qwen2.5-1.5B-Instruct'
            and bool(getattr(config, 'load_in_4bit', False))
            and getattr(config, 'bnb_4bit_quant_type', None) == 'nf4'
            and bool(getattr(config, 'bnb_4bit_use_double_quant', False))
            and getattr(config, 'bnb_4bit_compute_dtype', None) == torch.float16)


def install():
    import bitsandbytes as bnb
    import peft
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained.__func__
    if getattr(original, '_solution_cpu_streaming', False):
        return

    @functools.wraps(original)
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        name = pretrained_model_name_or_path
        if not _eligible_load(name, kwargs):
            return original(cls, name, *args, **kwargs)
        device = torch.device('cuda', torch.cuda.current_device())
        # Placement is an implementation detail of our allowed offload. The
        # exact original quantization config is passed through unmodified.
        offload_kwargs = dict(kwargs, device_map={'': 'cpu'})
        with _LOAD_LOCK:
            native_quantize = bnb.nn.Params4bit._quantize
            owner = threading.get_ident()

            @functools.wraps(native_quantize)
            def quantize(parameter, target):
                if threading.get_ident() != owner:
                    return native_quantize(parameter, target)
                # CPU quantization has a different backend. Always use the
                # exact native CUDA quantizer used by the original task.
                result = native_quantize(parameter, device)
                result.data = result.data.detach().cpu().pin_memory()
                result.quant_state.to(torch.device('cpu'))
                _pin_state(result.quant_state)
                result.quant_state._solution_cpu_streamed = True
                result.quant_state._solution_execution_device = device
                if getattr(result, 'module', None) is not None:
                    result.module.quant_state = result.quant_state
                return result

            bnb.nn.Params4bit._quantize = quantize
            try:
                model = original(cls, name, *args, **offload_kwargs)
            finally:
                bnb.nn.Params4bit._quantize = native_quantize
        model.__dict__['_solution_cpu_streamed'] = True
        model.__dict__['_solution_execution_device'] = device
        return model

    from_pretrained._solution_cpu_streaming = True
    AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
    native_peft = peft.get_peft_model

    @functools.wraps(native_peft)
    def get_peft_model(*args, **kwargs):
        return prepare_streamed_model(native_peft(*args, **kwargs))

    peft.get_peft_model = get_peft_model

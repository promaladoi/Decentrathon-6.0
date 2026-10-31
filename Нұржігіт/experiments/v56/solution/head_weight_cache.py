"""One in-process FP16 cast of a frozen private-training LM-head weight.

The caller selects this only inside the existing private training path. The
FP32 Parameter, optimizer, state_dict and public model forward are unchanged.
This caches the same ordinary ``weight.to(torch.float16)`` tensor, not a new
quantization, model file or compiler artifact. It is never saved across runs.

Normal versioned mutation, load_state_dict, replacement and exposed cached-copy
mutation invalidate the slot. Unsupported ``.data`` in-place writes can bypass
PyTorch version counters; they must not be used on the fixed frozen weight.
Call clear_half_weight_cache before such an external edit. The cache retains
one half copy, so whole-run peak memory must still be measured before enabling.
"""

from dataclasses import dataclass
import threading
import weakref

import torch


@dataclass
class _Entry:
    source: object
    signature: tuple
    half: torch.Tensor
    half_signature: tuple


_entry = None
_cache_lock = threading.RLock()
_weak_reference = weakref.ref


def _tensor_signature(tensor):
    return (int(tensor._version), tensor.data_ptr(), tensor.device, tensor.dtype,
            tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            bool(tensor.requires_grad))


def _source_signature(weight):
    # Cross-stream reuse would add a hidden dependency on the stream that
    # produced the cast. A different stream gets a fresh cast on that stream.
    stream = (torch.cuda.current_stream(weight.device).cuda_stream
              if weight.is_cuda else None)
    return (id(weight), *_tensor_signature(weight), stream)


def _source_expired(source):
    global _entry
    with _cache_lock:
        if _entry is not None and _entry.source is source:
            _entry = None


def clear_half_weight_cache(weight=None):
    """Release the slot, optionally only if it belongs to one exact source."""
    global _entry
    with _cache_lock:
        if _entry is not None and (weight is None or _entry.source() is weight):
            _entry = None


def get_half_weight(weight):
    """Return a cast equal to ``weight.to(float16)`` for a frozen 2D weight.

    A cache miss always allocates a new copy. Old copies are never modified,
    so autograd graphs retained from an earlier weight version remain valid.
    Trainable or unsupported inputs keep the ordinary torch/autograd path.
    """
    global _entry
    if not isinstance(weight, torch.Tensor):
        raise TypeError("LM-head weight must be a tensor")
    if (weight.requires_grad or weight.dtype != torch.float32 or weight.ndim != 2
            or weight.layout != torch.strided or weight.device.type not in ("cpu", "cuda")
            or weight.numel() == 0):
        clear_half_weight_cache()
        return weight.to(torch.float16)
    with _cache_lock:
        try:
            signature = _source_signature(weight)
            if (_entry is not None and _entry.source() is weight
                    and _entry.signature == signature
                    and _tensor_signature(_entry.half) == _entry.half_signature):
                return _entry.half
            reference = _weak_reference(weight, _source_expired)
        except (RuntimeError, TypeError, AttributeError):
            # Inference tensors may not expose version counters, and an
            # unusual tensor implementation may not support weak references.
            _entry = None
            return weight.to(torch.float16)
        # A surrounding inference_mode must not make this copy unsuitable for
        # saving in a later training graph. Frozen casts require no autograd.
        with torch.inference_mode(False), torch.no_grad():
            half = weight.to(torch.float16)
        _entry = _Entry(reference, signature, half, _tensor_signature(half))
        return half

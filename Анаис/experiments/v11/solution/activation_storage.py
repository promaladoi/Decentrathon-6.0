"""Run-local, lossless CPU storage for tensors saved by private training.

This is our own saved-tensor offload, not Soup's activation_offloading switch.
Arithmetic, RNG, gradients and tensor values remain the same. Model parameter
storage stays on its existing device. Large saved activations use pinned RAM;
the native stream orders D2H, the forward, H2D and its consumer.

Version witnesses keep native mutation detection without retaining GPU storage:
detach shares a version counter; assigning data on that independent detached
wrapper preserves its counter while replacing only the witness's storage.
Pinned torch2.11 sources: torch/csrc/autograd/variable.cpp, VariableHooks::set_data
and at::Tensor::detach. The original tensor is never assigned or modified.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import weakref

import torch


MIN_OFFLOAD_BYTES = 256 * 1024
MAX_VIEW_SPAN_RATIO = 4


def _storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def _copyable_layout(tensor):
    """Positive nonoverlapping strides, with bounded holes in their span."""
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    if any(stride < 0 for stride in strides):
        return None
    extent = 1
    for stride, size in sorted((s, n) for s, n in zip(strides, shape) if n > 1):
        if stride < extent:
            return None
        extent += (size - 1) * stride
    if extent > MAX_VIEW_SPAN_RATIO * tensor.numel():
        return None
    return shape, strides


@dataclass
class _Kept:
    tensor: torch.Tensor
    version: object


@dataclass
class _Offloaded:
    host: torch.Tensor
    device: torch.device
    witness: torch.Tensor
    version: int
    ready: object
    source: object


class ActivationStorage:
    """One forward's pack/unpack owner; no cache survives its autograd graph."""

    def __init__(self, model=None, *, minimum_bytes=MIN_OFFLOAD_BYTES):
        self.minimum_bytes = minimum_bytes
        self.protected = set()
        if model is not None:
            self.protected.update(_storage_key(t) for t in model.parameters()
                                  if t.layout == torch.strided)
            self.protected.update(_storage_key(t) for t in model.buffers()
                                  if t.layout == torch.strided)
        self._empty = {}
        self._copies = weakref.WeakValueDictionary()
        self.stats = {'offloaded_tensors': 0, 'offloaded_bytes': 0,
                      'reused_copies': 0, 'kept_tensors': 0, 'restored_tensors': 0}

    def _witness(self, tensor):
        empty = self._empty.get(tensor.dtype)
        if empty is None:
            empty = torch.empty(0, device='cpu', dtype=tensor.dtype)
            self._empty[tensor.dtype] = empty
        witness = tensor.detach()
        witness.data = empty
        return witness

    def pack(self, tensor):
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        if (not tensor.is_cuda or tensor.layout != torch.strided
                or tensor.numel() * tensor.element_size() < self.minimum_bytes
                or _storage_key(tensor) in self.protected or version is None):
            self.stats['kept_tensors'] += 1
            return _Kept(tensor.detach(), version)
        layout = _copyable_layout(tensor)
        if layout is None:
            self.stats['kept_tensors'] += 1
            return _Kept(tensor.detach(), version)
        key = (id(tensor), version, tensor.data_ptr(), tuple(tensor.shape),
               tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)
        previous = self._copies.get(key)
        if previous is not None and previous.source() is tensor:
            self.stats['reused_copies'] += 1
            return previous
        shape, strides = layout
        host = torch.empty_strided(shape, strides, dtype=tensor.dtype,
                                   device='cpu', pin_memory=True)
        stream = torch.cuda.current_stream(tensor.device)
        host.copy_(tensor, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(stream)
        packed = _Offloaded(host, tensor.device, self._witness(tensor), version,
                            ready, weakref.ref(tensor))
        self._copies[key] = packed
        self.stats['offloaded_tensors'] += 1
        self.stats['offloaded_bytes'] += tensor.numel() * tensor.element_size()
        return packed

    def unpack(self, packed):
        if isinstance(packed, _Kept):
            if packed.version is not None and packed.tensor._version != packed.version:
                raise RuntimeError('a tensor saved for backward was modified inplace')
            return packed.tensor
        if packed.witness._version != packed.version:
            raise RuntimeError('a tensor saved for backward was modified inplace')
        stream = torch.cuda.current_stream(packed.device)
        stream.wait_event(packed.ready)
        data = torch.empty_strided(packed.host.shape, packed.host.stride(),
                                  dtype=packed.host.dtype, device=packed.device)
        data.copy_(packed.host, non_blocking=True)
        # Share the original version counter, not its former GPU allocation.
        restored = packed.witness.detach()
        restored.data = data
        self.stats['restored_tensors'] += 1
        return restored

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


@contextmanager
def activation_storage(model):
    """Enter only after the private forward's normal eligibility guards pass."""
    manager = ActivationStorage(model)
    model.__dict__['_solution_activation_storage_stats'] = manager.stats
    with manager.hooks():
        yield manager

"""Bounded, process-local reuse of decoded frozen NF4 projection weights.

This trades some of the saved VRAM for fewer repeated decodes/allocations. The
packed model parameters and quantization settings remain untouched. No decoded
weights, compilation artifacts or adapters are carried between measured runs.
"""
import math
import weakref

import torch

MAX_BYTES = 2_800_000_000
MIN_ELEMENTS = 262_144


def _snapshot(packed, state, dtype):
    tensors = [packed]
    tensors += [getattr(state, name, None) for name in ('code', 'absmax', 'offset')]
    nested = bool(getattr(state, 'nested', False))
    second = getattr(state, 'state2', None) if nested else None
    if second is not None:
        tensors += [getattr(second, name, None) for name in ('code', 'absmax')]
    tensors = [t for t in tensors if torch.is_tensor(t)]
    if any(t.requires_grad for t in tensors):
        return None
    try:
        metadata = tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()),
                          t.dtype, t.device) for t in tensors)
    except RuntimeError:  # inference tensors cannot provide mutation versions
        return None
    scalar_offset = getattr(state, 'offset', None)
    scalar_offset = scalar_offset if isinstance(scalar_offset, (int, float)) else None
    layout = (tuple(state.shape), getattr(state, 'quant_type', None), state.blocksize,
              nested, getattr(second, 'blocksize', None), scalar_offset, dtype)
    return layout, metadata, tensors


class FrozenWeightCache:
    """Mutation-aware tensor cache; routing to eligible CUDA matrices is separate."""
    def __init__(self, decoder, max_bytes=MAX_BYTES):
        self.decoder, self.max_bytes = decoder, max_bytes
        self.entries, self.bytes = {}, 0

    def _drop(self, key, owner=None):
        entry = self.entries.get(key)
        if entry is not None and (owner is None or entry['owner'] is owner):
            self.bytes -= entry['bytes']
            del self.entries[key]

    def get(self, packed, state, dtype):
        if packed.requires_grad:
            return self.decoder(packed, state, dtype)
        snapshot = _snapshot(packed, state, dtype)
        if snapshot is None:
            return self.decoder(packed, state, dtype)
        layout, metadata, tensors = snapshot
        key = id(packed)
        entry = self.entries.get(key)
        if (entry is not None and entry['owner']() is packed and
                entry['signature'] == (layout, metadata) and
                all(ref() is t for ref, t in zip(entry['refs'], tensors))):
            return entry['value']
        self._drop(key)
        value = self.decoder(packed, state, dtype)
        size = value.numel()*value.element_size()
        if (size > self.max_bytes-self.bytes or
                value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()):
            return value
        owner = weakref.ref(packed, lambda ref: self._drop(key, ref))
        self.entries[key] = {'owner': owner, 'refs': [weakref.ref(t) for t in tensors],
                             'signature': (layout, metadata), 'value': value, 'bytes': size}
        self.bytes += size
        return value


def cacheable_size(shape,min_elements,max_elements=None):
    count=math.prod(shape)
    return count>=min_elements and (max_elements is None or count<max_elements)


def patch_frozen_weight_cache(min_elements=MIN_ELEMENTS,max_elements=None,max_bytes=MAX_BYTES):
    from .team_vladimir import mlp, nf4, unpad
    original = nf4.dequantize_nf4
    if hasattr(original, '_frozen_weight_cache'):
        return original._frozen_weight_cache
    cache = FrozenWeightCache(original,max_bytes=max_bytes)

    def decode(packed, state, dtype):
        if (packed.is_cuda and not packed.requires_grad and packed.is_contiguous() and
                dtype == torch.float16 and nf4.supported(state) and
                cacheable_size(state.shape,min_elements,max_elements)):
            return cache.get(packed, state, dtype)
        return original(packed, state, dtype)

    decode._frozen_weight_cache = cache
    nf4.dequantize_nf4 = decode
    mlp.dequantize_nf4 = decode
    unpad.dequantize_nf4 = decode
    return cache

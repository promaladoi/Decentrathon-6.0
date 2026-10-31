"""Bounded reuse for internal read-only consumers of frozen NF4 matrices.

Adapted from this team's v39 FrozenWeightCache, commit 0a05a86. The current
canonical public decoder still returns fresh tensors. Only internal native
linear/matmul consumers use this slot. Every process starts empty; decoding,
allocation and validation all remain inside the measured training command.
"""
import weakref
import torch

MAX_BYTES = 2_400_000_000


def _stream(packed):
    return torch.cuda.current_stream(packed.device).cuda_stream if packed.is_cuda else None


def _layout(state, dtype, stream):
    nested = bool(getattr(state,'nested',False))
    second = getattr(state,'state2',None) if nested else None
    offset = getattr(state,'offset',None)
    return (tuple(state.shape),getattr(state,'dtype',None),getattr(state,'quant_type',None),
            state.blocksize,nested,getattr(second,'blocksize',None),
            getattr(second,'nested',None),getattr(second,'dtype',None),
            offset if isinstance(offset,(int,float)) else None,dtype,stream)


def _tensors(packed,state):
    values=[packed,getattr(state,'code',None),getattr(state,'absmax',None),getattr(state,'offset',None)]
    second=getattr(state,'state2',None) if getattr(state,'nested',False) else None
    if second is not None:
        values.extend((getattr(second,'code',None),getattr(second,'absmax',None)))
    return [t for t in values if isinstance(t,torch.Tensor)]


def _record(tensor):
    return (tensor._version,tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),
            tensor.dtype,tensor.device)


class FrozenWeightCache:
    """Versioned, stream-specific frozen data; overflow decodes without retention.

    Ordinary versioned mutation and data replacement invalidate entries.
    Unversioned external .data in-place writes on these frozen parameters
    are outside this cache's contract; clear it before such an edit.
    """
    def __init__(self,decoder,max_bytes=MAX_BYTES):
        self.decoder,self.max_bytes=decoder,max_bytes
        self.entries,self.bytes={},0

    def clear(self):
        self.entries.clear()
        self.bytes=0

    def _drop(self,key,owner=None):
        entry=self.entries.get(key)
        if entry is not None and (owner is None or entry['owner'] is owner):
            self.bytes-=entry['bytes']
            del self.entries[key]

    def get(self,packed,state,dtype):
        key=id(packed)
        tensors=_tensors(packed,state)
        try:
            if any(t.requires_grad for t in tensors):
                self._drop(key)
                return self.decoder(packed,state,dtype)
            layout=_layout(state,dtype,_stream(packed))
            entry=self.entries.get(key)
            if (entry is not None and entry['owner']() is packed and entry['layout']==layout
                    and len(tensors)==len(entry['refs'])
                    and all(ref() is t and _record(t)==record
                            for t,ref,record in zip(tensors,entry['refs'],entry['records']))
                    and _record(entry['value'])==entry['value_record']):
                return entry['value']
            records=tuple(_record(t) for t in tensors)
        except RuntimeError:  # inference tensors do not expose mutation versions
            self._drop(key)
            return self.decoder(packed,state,dtype)
        self._drop(key)
        value=self.decoder(packed,state,dtype)
        size=value.numel()*value.element_size()
        if (size>self.max_bytes-self.bytes or value.requires_grad
                or value.untyped_storage().data_ptr()==packed.untyped_storage().data_ptr()):
            return value
        try:
            value_record=_record(value)
        except RuntimeError:
            return value
        owner=weakref.ref(packed,lambda ref:self._drop(key,ref))
        self.entries[key]=dict(owner=owner,refs=tuple(weakref.ref(t) for t in tensors),
            records=records,layout=layout,value=value,value_record=value_record,bytes=size)
        self.bytes+=size
        return value

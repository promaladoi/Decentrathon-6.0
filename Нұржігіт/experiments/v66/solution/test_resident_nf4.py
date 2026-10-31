"""Frozen-input mutation, bounded storage and exact native GPU qualification."""
import gc
from types import SimpleNamespace
import weakref
import pytest
import torch


def _setup(nested=False,max_bytes=1024):
    from solution.resident_nf4 import FrozenWeightCache
    p=torch.arange(16,dtype=torch.uint8)
    s=SimpleNamespace(shape=(4,4),dtype=torch.float16,quant_type='nf4',blocksize=64,
        nested=nested,code=torch.arange(16).float(),absmax=torch.tensor([.25]),offset=torch.tensor(.125),
        state2=SimpleNamespace(blocksize=256,nested=False,dtype=torch.float32,
                              code=torch.ones(2),absmax=torch.tensor([.5])))
    calls=[]
    def decode(packed,state,dtype):
        calls.append(1)
        scale=state.absmax[0]*(state.state2.absmax[0] if state.nested else 1)
        return (packed.float()*scale+state.offset).to(dtype).reshape(state.shape)
    return p,s,FrozenWeightCache(decode,max_bytes),calls


def test_unchanged_values_reuse_one_allocation_and_clear_refunds_budget():
    p,s,cache,calls=_setup()
    first=cache.get(p,s,torch.float16)
    assert cache.get(p,s,torch.float16) is first
    assert len(calls)==1 and cache.bytes==32
    cache.clear()
    assert not cache.entries and cache.bytes==0
    assert cache.get(p,s,torch.float16) is not first


@pytest.mark.parametrize('change',['packed','scale','offset','replace_scale','nested_scale','code','dtype','shape','output'])
def test_mutation_replacement_or_layout_change_invalidates(change):
    p,s,cache,calls=_setup(nested=True)
    first=cache.get(p,s,torch.float16)
    if change=='packed': p.add_(1)
    elif change=='scale': s.absmax.mul_(2)
    elif change=='offset': s.offset.add_(.25)
    elif change=='replace_scale': s.absmax=s.absmax.clone()*2
    elif change=='nested_scale': s.state2.absmax.mul_(2)
    elif change=='code': s.code.add_(1)
    elif change=='dtype': s.dtype=torch.float32
    elif change=='shape': s.shape=(2,8)
    else: first.add_(1)
    result=cache.get(p,s,torch.float16)
    assert result is not first and len(calls)==2 and cache.bytes==32
    expected=(p.float()*s.absmax[0]*s.state2.absmax[0]+s.offset).half().reshape(s.shape)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)


def test_data_replacement_without_version_change_invalidates():
    p,s,cache,calls=_setup()
    first=cache.get(p,s,torch.float16)
    version=p._version
    p.data=p.clone().add_(1)
    assert p._version==version
    assert cache.get(p,s,torch.float16) is not first and len(calls)==2


def test_budget_never_evicts_valid_first_entry_or_overflows():
    p,s,cache,calls=_setup(max_bytes=32)
    first=cache.get(p,s,torch.float16)
    other=p.clone()
    a=cache.get(other,s,torch.float16)
    b=cache.get(other,s,torch.float16)
    assert a is not b and cache.bytes==32
    assert cache.get(p,s,torch.float16) is first


def test_owner_deletion_releases_entry_and_aliasing_decoder_is_not_retained():
    from solution.resident_nf4 import FrozenWeightCache
    p,s,cache,_=_setup()
    value=cache.get(p,s,torch.float16)
    owner=weakref.ref(p)
    del p
    gc.collect()
    assert owner() is None and cache.bytes==0 and not cache.entries
    p=torch.arange(17,dtype=torch.float16)
    aliases=FrozenWeightCache(lambda p,s,d:p[1:].view(4,4))
    assert aliases.get(p,s,torch.float16).shape==(4,4)
    assert aliases.bytes==0 and not aliases.entries


@pytest.mark.parametrize('kind',['parameter','state','inference'])
def test_trainable_or_unversioned_inputs_are_never_retained(kind):
    p,s,cache,calls=_setup()
    if kind=='parameter': p=p.float().requires_grad_()
    elif kind=='state': s.absmax.requires_grad_()
    else:
        with torch.inference_mode(): p=torch.arange(16,dtype=torch.uint8)
    cache.get(p,s,torch.float16)
    cache.get(p,s,torch.float16)
    assert cache.bytes==0 and len(calls)==2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs CUDA')
@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('state_dtype',[torch.float16,torch.float32])
def test_internal_forward_backward_exact_and_public_decoder_stays_fresh(monkeypatch,nested,state_dtype):
    from solution import nf4
    import bitsandbytes.functional as bnb
    monkeypatch.setattr(nf4,'_resident_cache',None)
    torch.manual_seed(650)
    packed,state=bnb.quantize_4bit(torch.randn((256,512),device='cuda',dtype=state_dtype),
        blocksize=64,compress_statistics=nested,quant_type='nf4')
    x=torch.randn((512,512),device='cuda',dtype=torch.float16)
    grad=torch.randn((512,256),device='cuda',dtype=torch.float16)
    expected=nf4._linear_forward(x,packed,None,state)
    expected_back=torch.matmul(grad,nf4.dequantize_nf4(packed,state,torch.float16))
    cache=nf4.enable_resident_weights(max_bytes=1_048_576)
    actual=nf4._linear_forward(x,packed,None,state)
    actual_back=torch.matmul(grad,nf4._readonly_weight(packed,state,torch.float16))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual_back,expected_back,rtol=0,atol=0)
    if state_dtype==torch.float16:
        assert nf4._readonly_weight(packed,state,torch.float16) is nf4._readonly_weight(packed,state,torch.float16)
        assert cache.bytes==256*512*2
    else: assert cache.bytes==0
    first=nf4.dequantize_nf4(packed,state,torch.float16)
    second=nf4.dequantize_nf4(packed,state,torch.float16)
    assert first.data_ptr()!=second.data_ptr()
    first.zero_()
    torch.testing.assert_close(second,bnb.dequantize_4bit(packed,state).to(torch.float16),rtol=0,atol=0)
    packed.add_(1)
    torch.testing.assert_close(nf4._readonly_weight(packed,state,torch.float16),
        bnb.dequantize_4bit(packed,state).to(torch.float16),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs CUDA')
def test_another_cuda_stream_gets_an_independent_decode(monkeypatch):
    from solution import nf4
    import bitsandbytes.functional as bnb
    monkeypatch.setattr(nf4,'_resident_cache',None)
    packed,state=bnb.quantize_4bit(torch.randn((256,512),device='cuda',dtype=torch.float16),
        blocksize=64,compress_statistics=True,quant_type='nf4')
    nf4.enable_resident_weights(max_bytes=1_048_576)
    torch.cuda.synchronize()
    one,two=torch.cuda.Stream(),torch.cuda.Stream()
    with torch.cuda.stream(one): first=nf4._readonly_weight(packed,state,torch.float16)
    with torch.cuda.stream(two): second=nf4._readonly_weight(packed,state,torch.float16)
    torch.cuda.synchronize()
    assert first.data_ptr()!=second.data_ptr()
    torch.testing.assert_close(first,second,rtol=0,atol=0)

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch


@pytest.mark.parametrize('shape,expected',[
    ((512,511),False),((512,512),True),((1536,1536),True),
    ((256,1536),True),((2048,2048),False),((8960,1536),False),
])
def test_attention_only_cache_size_boundaries(shape,expected):
    from .frozen_weight_cache import cacheable_size
    assert cacheable_size(shape,262_144,4_194_304) is expected


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs CUDA')
def test_actual_small_only_routing_retains_only_attention_matrices(monkeypatch):
    import bitsandbytes.functional as bnb
    from .team_vladimir import nf4,mlp,unpad
    from .frozen_weight_cache import patch_frozen_weight_cache
    # Restore module bindings after this isolated policy test.
    for module in (nf4,mlp,unpad):
        monkeypatch.setattr(module,'dequantize_nf4',nf4.dequantize_nf4)
    cache=patch_frozen_weight_cache(max_elements=4_194_304,max_bytes=400_000_000)
    for shape,retain in [((1024,512),True),((8960,1536),False)]:
        x=torch.randn(shape,device='cuda',dtype=torch.float16)
        packed,state=bnb.quantize_4bit(x,quant_type='nf4',compress_statistics=True)
        a=nf4.dequantize_nf4(packed,state,torch.float16)
        b=nf4.dequantize_nf4(packed,state,torch.float16)
        assert (a is b) is retain
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert cache.bytes<=400_000_000

from solution.frozen_weight_cache import FrozenWeightCache


def setup(nested=False, max_bytes=1024):
    packed = torch.arange(16, dtype=torch.uint8)
    state = SimpleNamespace(shape=(4, 4), quant_type='nf4', blocksize=64,
                            nested=nested, code=torch.arange(16).float(),
                            absmax=torch.tensor([.25]), offset=torch.tensor(.125),
                            state2=SimpleNamespace(blocksize=256, code=torch.ones(2),
                                                   absmax=torch.tensor([.5])))
    calls = []

    def decode(p, s, d):
        calls.append(1)
        scale = s.absmax[0]
        if s.nested:
            scale = scale*s.state2.absmax[0]
        return (p.float()*scale+s.offset).to(d).view(s.shape)

    return packed, state, FrozenWeightCache(decode, max_bytes), calls


def test_unchanged_frozen_values_reuse_same_decoded_tensor():
    p, s, cache, calls = setup()
    a = cache.get(p, s, torch.float16)
    b = cache.get(p, s, torch.float16)
    assert a is b and len(calls) == 1 and cache.bytes == 32
    torch.testing.assert_close(a, (p.float()*.25+.125).half().view(4, 4), rtol=0, atol=0)


@pytest.mark.parametrize('change', ['packed', 'scale', 'offset', 'replace_scale', 'nested_scale'])
def test_value_or_state_mutations_invalidate(change):
    p, s, cache, calls = setup(nested=True)
    before = cache.get(p, s, torch.float16)
    if change == 'packed':
        p.add_(1)
    elif change == 'scale':
        s.absmax.mul_(2)
    elif change == 'offset':
        s.offset.add_(.25)
    elif change == 'replace_scale':
        s.absmax = s.absmax.clone()*2
    else:
        s.state2.absmax.mul_(2)
    actual = cache.get(p, s, torch.float16)
    assert actual is not before and len(calls) == 2 and cache.bytes == 32
    expected = (p.float()*s.absmax[0]*s.state2.absmax[0]+s.offset).half().view(s.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dtype_change_refunds_previous_resident_size():
    p, s, cache, calls = setup()
    cache.get(p, s, torch.float32)
    assert cache.bytes == 64
    a = cache.get(p, s, torch.float16)
    assert a.dtype == torch.float16 and cache.bytes == 32 and len(calls) == 2


def test_data_replacement_without_version_increment_invalidates_hit():
    p, s, cache, calls = setup()
    before = cache.get(p, s, torch.float16)
    version = p._version
    p.data = p.data.clone().add_(1)
    assert p._version == version
    after = cache.get(p, s, torch.float16)
    assert after is not before and len(calls) == 2
    torch.testing.assert_close(after, (p.float()*.25+.125).half().view(4,4), rtol=0, atol=0)


def test_metadata_change_and_trainable_state_invalidate_previously_cached_hit():
    p, s, cache, calls = setup()
    before = cache.get(p, s, torch.float16)
    s.shape = (2, 8)
    after = cache.get(p, s, torch.float16)
    assert after is not before and after.shape == (2,8) and len(calls) == 2
    s.absmax.requires_grad_()
    cache.get(p, s, torch.float16)
    assert len(calls) == 3


def test_budget_limit_decodes_without_retaining():
    p, s, cache, calls = setup(max_bytes=31)
    a, b = cache.get(p, s, torch.float16), cache.get(p, s, torch.float16)
    assert a is not b and cache.bytes == 0 and not cache.entries and len(calls) == 2


def test_second_weight_cannot_exceed_budget_or_evict_valid_first():
    p, s, cache, calls = setup(max_bytes=32)
    first = cache.get(p, s, torch.float16)
    other = p.clone()
    cache.get(other, s, torch.float16)
    assert cache.get(p, s, torch.float16) is first and cache.bytes == 32 and len(calls) == 2


def test_parameter_deletion_releases_resident_entry():
    p, s, cache, _ = setup()
    value = cache.get(p, s, torch.float16)
    ref = weakref.ref(p)
    del p
    gc.collect()
    assert ref() is None and cache.bytes == 0 and not cache.entries
    assert value.shape == (4, 4)


def test_trainable_input_is_never_cached():
    p, s, cache, calls = setup()
    p = p.float().requires_grad_()
    cache.get(p, s, torch.float16)
    cache.get(p, s, torch.float16)
    assert cache.bytes == 0 and len(calls) == 2


def test_inference_tensor_without_mutation_version_is_not_cached():
    _, s, cache, calls = setup()
    with torch.inference_mode():
        p = torch.arange(16, dtype=torch.uint8)
    cache.get(p, s, torch.float16)
    cache.get(p, s, torch.float16)
    assert cache.bytes == 0 and len(calls) == 2


def test_trainable_quantization_state_is_not_cached():
    p, s, cache, calls = setup()
    s.absmax.requires_grad_()
    cache.get(p, s, torch.float16)
    cache.get(p, s, torch.float16)
    assert cache.bytes == 0 and len(calls) == 2


def test_aliasing_decoder_output_cannot_keep_owner_alive_in_cache():
    _, s, _, _ = setup()
    p = torch.arange(17, dtype=torch.float16)
    cache = FrozenWeightCache(lambda p, s, d: p[1:].view(4, 4))
    result = cache.get(p, s, torch.float16)
    assert result.data_ptr() != p.data_ptr()
    assert result.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
    assert cache.bytes == 0 and not cache.entries


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('shape', [(8960, 1536), (1536, 8960)])
@pytest.mark.parametrize('nested', [False, True])
def test_actual_nf4_cache_reuse_and_invalidation_exact(shape, nested):
    import bitsandbytes.functional as bnb
    from solution.team_vladimir.nf4 import dequantize_nf4
    weight = torch.randn(shape, device='cuda', dtype=torch.float16)
    packed, state = bnb.quantize_4bit(weight, quant_type='nf4', compress_statistics=nested)
    cache = FrozenWeightCache(dequantize_nf4)
    expected = dequantize_nf4(packed, state, torch.float16)
    a = cache.get(packed, state, torch.float16)
    assert cache.get(packed, state, torch.float16) is a
    torch.testing.assert_close(a, expected, rtol=0, atol=0)
    packed.add_(1)
    b = cache.get(packed, state, torch.float16)
    assert b is not a
    torch.testing.assert_close(b, dequantize_nf4(packed, state, torch.float16), rtol=0, atol=0)

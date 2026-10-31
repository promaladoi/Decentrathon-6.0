"""Merey's bounded adaptation of Anais's exact run-local NF4 decode cache."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


def test_attention_cache_policy_boundaries_without_gpu():
    """Execute the actual allocation policy, without pretending to test CUDA math."""
    tree = ast.parse(Path(__file__).with_name('nf4.py').read_bytes())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_cache_shape_supported']
    torch = SimpleNamespace(float16='f16', float32='f32', Size=tuple)
    namespace = {'torch': torch, 'DECODE_CACHE_MAX_DIM': 1536}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'nf4.py', 'exec'), namespace)
    eligible = namespace['_cache_shape_supported']
    for shape in ((1536, 1536), (256, 1536), (128, 256)):
        assert eligible(SimpleNamespace(shape=shape, dtype='f16'), 'f16')
    for shape in ((8960, 1536), (1536, 8960), (0, 1536), (1536,), ('1536', 1536)):
        assert not eligible(SimpleNamespace(shape=shape, dtype='f16'), 'f16')
    assert not eligible(SimpleNamespace(shape=(1536, 1536), dtype='f32'), 'f16')
    assert not eligible(SimpleNamespace(shape=(1536, 1536), dtype='f16'), 'f32')


def _cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return torch


def _quantized(torch, shape=(128, 256), nested=True):
    import bitsandbytes.functional as bnbf
    return bnbf.quantize_4bit(torch.randn(shape, device='cuda', dtype=torch.float16),
                             compress_statistics=nested, quant_type='nf4')


@pytest.mark.parametrize('source', ['packed', 'absmax', 'code', 'second_absmax',
                                   'second_code', 'offset'])
def test_each_source_mutation_invalidates_exact_cached_decode(source):
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    first = nf4._decode(packed, state, torch.float16)
    tensor = {'packed': packed, 'absmax': state.absmax, 'code': state.code,
              'second_absmax': state.state2.absmax, 'second_code': state.state2.code,
              'offset': state.offset}[source]
    tensor.add_(1 if tensor.dtype == torch.uint8 else 0.001)
    changed = nf4._decode(packed, state, torch.float16)
    assert changed is not first
    torch.testing.assert_close(changed, nf4._decode_uncached(packed, state, torch.float16),
                               rtol=0, atol=0)


def test_returned_tensor_mutation_invalidates_without_overwriting_old_graph():
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    first = nf4._decode(packed, state, torch.float16)
    first.zero_()
    changed = nf4._decode(packed, state, torch.float16)
    assert changed is not first and torch.count_nonzero(first) == 0
    torch.testing.assert_close(changed, nf4._decode_uncached(packed, state, torch.float16),
                               rtol=0, atol=0)


def test_stream_change_decodes_on_current_stream_without_cross_stream_reuse():
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    first = nf4._decode(packed, state, torch.float16)
    current = torch.cuda.current_stream()
    other = torch.cuda.Stream()
    other.wait_stream(current)
    with torch.cuda.stream(other):
        changed = nf4._decode(packed, state, torch.float16)
        assert changed is not first
        assert nf4._decode(packed, state, torch.float16) is changed
    current.wait_stream(other)
    restored = nf4._decode(packed, state, torch.float16)
    assert restored is not changed
    torch.testing.assert_close(restored, first, rtol=0, atol=0)


def test_global_cap_never_evicts_hot_entries_and_reclaims_dead_states(monkeypatch):
    torch = _cuda()
    from solution import nf4
    gc.collect()
    before = nf4._decode_cache_bytes
    first_packed, first_state = _quantized(torch)
    second_packed, second_state = _quantized(torch)
    size = 128 * 256 * 2
    monkeypatch.setattr(nf4, 'DECODE_CACHE_MAX_BYTES', before + size)
    first = nf4._decode(first_packed, first_state, torch.float16)
    assert nf4._decode_cache_bytes == before + size
    second = nf4._decode(second_packed, second_state, torch.float16)
    assert getattr(second_state, '_anais_decode_cache', None) is None
    assert nf4._decode(first_packed, first_state, torch.float16) is first
    assert nf4._decode_cache_bytes == before + size
    old_state = weakref.ref(first_state)
    del first_state, first_packed
    gc.collect()
    assert old_state() is None
    assert nf4._decode_cache_bytes == before
    # A user-held old dense tensor remains valid but does not pin its model.
    assert first.shape == second.shape
    cached_second = nf4._decode(second_packed, second_state, torch.float16)
    assert nf4._decode(second_packed, second_state, torch.float16) is cached_second
    assert nf4._decode_cache_bytes == before + size


def test_cache_does_not_create_parameter_quant_state_reference_cycle():
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    # Params4bit normally points back to its QuantState. Reproduce that edge.
    packed.quant_state = state
    before = nf4._decode_cache_bytes
    dense = nf4._decode(packed, state, torch.float16)
    source_ref, state_ref = weakref.ref(packed), weakref.ref(state)
    del packed
    assert source_ref() is None
    del state
    assert state_ref() is None
    assert nf4._decode_cache_bytes == before
    assert dense.shape == (128, 256)


@pytest.mark.parametrize('shape', [(8960, 1536), (1536, 8960)])
def test_large_mlp_weights_never_enter_cache(shape):
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch, shape)
    before = nf4._decode_cache_bytes
    result = nf4._decode(packed, state, torch.float16)
    assert getattr(state, '_anais_decode_cache', None) is None
    assert nf4._decode_cache_bytes == before
    torch.testing.assert_close(result, nf4._decode_uncached(packed, state, torch.float16),
                               rtol=0, atol=0)


def test_inference_call_leaves_cache_usable_for_later_training():
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    with torch.inference_mode():
        cached = nf4._decode(packed, state, torch.float16)
    x = torch.randn(3, 256, device='cuda', dtype=torch.float16, requires_grad=True)
    out = torch.nn.functional.linear(x, nf4._decode(packed, state, torch.float16))
    out.sum().backward()
    assert x.grad is not None and not cached.is_inference()


def test_other_destination_dtype_drops_slot_and_preserves_raw_decode():
    torch = _cuda()
    from solution import nf4
    packed, state = _quantized(torch)
    nf4._decode(packed, state, torch.float16)
    assert state._anais_decode_cache is not None
    before = nf4._decode_cache_bytes
    decoded = nf4._decode(packed, state, torch.float32)
    assert state._anais_decode_cache is None
    assert nf4._decode_cache_bytes == before - 128 * 256 * 2
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, nf4._decode_uncached(packed, state, torch.float32),
                               rtol=0, atol=0)


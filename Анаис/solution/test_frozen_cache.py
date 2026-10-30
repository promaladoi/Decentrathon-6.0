"""Correctness and invalidation of the Anais run-local frozen-weight cache."""
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('nested', [True, False])
def test_frozen_mlp_cache_invalidation(nested):
    import bitsandbytes.functional as bnbf
    from solution.mlp import _weight
    torch.manual_seed(37)
    original = torch.randn(128, 256, device='cuda', dtype=torch.float16)
    packed, state = bnbf.quantize_4bit(original, compress_statistics=nested, quant_type='nf4')
    packed.quant_state = state
    layer = SimpleNamespace(weight=packed)
    first = _weight(layer, torch.float16)
    torch.testing.assert_close(first, bnbf.dequantize_4bit(packed, state), rtol=1e-3, atol=1e-5)
    assert _weight(layer, torch.float16).data_ptr() == first.data_ptr()
    # An in-place scale change must not reuse an old dequantized tensor.
    state.absmax.copy_(state.absmax.roll(1))
    second = _weight(layer, torch.float16)
    assert second.data_ptr() != first.data_ptr()
    torch.testing.assert_close(second, bnbf.dequantize_4bit(packed, state), rtol=1e-3, atol=1e-5)
    third = _weight(layer, torch.float32)
    assert third.dtype == torch.float32
    torch.testing.assert_close(third, bnbf.dequantize_4bit(packed, state).float(), rtol=1e-3, atol=1e-5)

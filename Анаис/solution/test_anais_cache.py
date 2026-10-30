import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('nested',[True,False])
def test_frozen_decode_cache_and_invalidation(nested):
    import bitsandbytes.functional as bnbf
    from solution.nf4 import _decode
    torch.manual_seed(151)
    packed,state=bnbf.quantize_4bit(
        torch.randn(128,256,device='cuda',dtype=torch.float16),
        compress_statistics=nested,quant_type='nf4')
    first=_decode(packed,state,torch.float16)
    torch.testing.assert_close(first,bnbf.dequantize_4bit(packed,state),rtol=1e-3,atol=1e-5)
    assert _decode(packed,state,torch.float16).data_ptr()==first.data_ptr()
    packed.bitwise_xor_(1)
    changed=_decode(packed,state,torch.float16)
    assert changed.data_ptr()!=first.data_ptr()
    torch.testing.assert_close(changed,bnbf.dequantize_4bit(packed,state),rtol=1e-3,atol=1e-5)
    state.absmax.copy_(state.absmax.roll(1))
    rescaled=_decode(packed,state,torch.float16)
    assert rescaled.data_ptr()!=changed.data_ptr()
    torch.testing.assert_close(rescaled,bnbf.dequantize_4bit(packed,state),rtol=1e-3,atol=1e-5)

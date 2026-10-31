"""Native-exact NF4 values across the larger launch geometry and fallbacks."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires a CUDA GPU')


@pytest.mark.parametrize('shape',[(1536,1536),(8960,1536)])
@pytest.mark.parametrize('weight_dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('output_dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('nested',[False,True])
def test_large_nf4_launch_preserves_native_values(shape,weight_dtype,output_dtype,nested):
    import bitsandbytes.functional as functional
    from solution.nf4 import dequantize_nf4
    torch.manual_seed(6301)
    weight=torch.randn(shape,device='cuda',dtype=weight_dtype)*.03
    packed,state=functional.quantize_4bit(weight,blocksize=64,
        quant_type='nf4',compress_statistics=nested)
    expected=functional.dequantize_4bit(packed,state).to(output_dtype)
    actual=dequantize_nf4(packed,state,output_dtype)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

"""Narrow vendor-dispatch candidate: native arithmetic and exact input gradients.

The forward follows the existing FP16 2e-3 absolute/relative tolerance from
test_kernels.py. Its FP16 MMA reduction can differ from the unfused vendor
GEMM. Existing bitwise tests are untouched. Original loss checks stay required.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('rows', [128, 512, 832, 1536])
@pytest.mark.parametrize('bias_enabled', [False, True])
def test_narrow_forward_and_exact_native_gradient(rows, bias_enabled):
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import matmul_4bit as native_matmul
    from solution.nf4 import MatMulNF4, _narrow_fused_supported
    torch.manual_seed(5601)
    weight = torch.randn(256, 1536, device='cuda', dtype=torch.float16)*.03
    packed, state = functional.quantize_4bit(weight, blocksize=64,
        compress_statistics=True, quant_type='nf4')
    bias = torch.randn(256, device='cuda', dtype=torch.float16)*.03 if bias_enabled else None
    x = torch.randn(rows, 1536, device='cuda', dtype=torch.float16, requires_grad=True)*.1
    grad = torch.randn(rows, 256, device='cuda', dtype=torch.float16)
    reference = native_matmul(x, packed, state, bias=bias)
    expected_dx, = torch.autograd.grad(reference, (x,), grad)
    actual = MatMulNF4.apply(x, packed, bias, state)
    actual_dx, = torch.autograd.grad(actual, (x,), grad)
    assert actual.dtype == reference.dtype == torch.float16
    assert _narrow_fused_supported(x, bias, state) == (torch.cuda.get_device_capability() == (7, 5))
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)


def test_narrow_dispatch_rejects_unmeasured_large_mlp_and_strided_inputs():
    import bitsandbytes.functional as functional
    from solution.nf4 import _narrow_fused_supported
    torch.manual_seed(5602)
    weight = torch.randn(512, 1536, device='cuda', dtype=torch.float16)
    _, state = functional.quantize_4bit(weight, quant_type='nf4', compress_statistics=True)
    x = torch.randn(512, 1536, device='cuda', dtype=torch.float16)
    assert not _narrow_fused_supported(x, None, state)
    state.shape = (256, 1536)
    assert not _narrow_fused_supported(x[:1].expand(512, -1), None, state)
    assert not _narrow_fused_supported(x.float(), None, state)
    assert not _narrow_fused_supported(x, torch.zeros(256, device='cuda'), state)

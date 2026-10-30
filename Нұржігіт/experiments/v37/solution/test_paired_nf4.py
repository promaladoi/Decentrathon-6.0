from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def _reference():
    from solution.team_vladimir import nf4
    return getattr(nf4.dequantize_nf4, '_paired_nf4_original', nf4.dequantize_nf4)


@pytest.mark.parametrize('n', [511, 512, 4097])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_all_nibbles_and_odd_tail_equal_original(n, nested, dtype):
    from solution.paired_nf4 import dequantize_pairs
    torch.manual_seed(139)
    # Includes all 256 bytes, all high/low nibbles, and a masked odd tail.
    packed = (torch.arange((n + 1) // 2, device='cuda') % 256).to(torch.uint8)
    blocks = (n + 63) // 64
    state = SimpleNamespace(shape=(1, n), quant_type='nf4', blocksize=64,
                            code=torch.linspace(-1, 1, 16, device='cuda'), nested=nested)
    if nested:
        state.absmax = torch.randint(256, (blocks,), device='cuda', dtype=torch.uint8)
        state.offset = torch.tensor(.125, device='cuda')
        state.state2 = SimpleNamespace(blocksize=256,
            code=torch.linspace(-.7, 1.3, 256, device='cuda'),
            absmax=torch.rand((blocks + 255) // 256, device='cuda')+.5)
    else:
        state.absmax = torch.rand(blocks, device='cuda')+.5
    expected = _reference()(packed, state, dtype)
    actual = dequantize_pairs(packed, state, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(256, 1536), (8960, 1536)])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_real_nf4_weights_equal_original(shape, nested, dtype):
    import bitsandbytes.functional as bnbf
    from solution.paired_nf4 import dequantize_pairs
    torch.manual_seed(149)
    weight = torch.randn(shape, device='cuda', dtype=torch.float16)
    packed, state = bnbf.quantize_4bit(weight, blocksize=64,
                                     compress_statistics=nested, quant_type='nf4')
    torch.testing.assert_close(dequantize_pairs(packed, state, dtype),
                               _reference()(packed, state, dtype), rtol=0, atol=0)


@pytest.mark.parametrize('nested', [False, True])
def test_frozen_linear_output_and_input_gradient_equal(nested):
    import bitsandbytes.functional as bnbf
    from solution.paired_nf4 import dequantize_pairs
    torch.manual_seed(151)
    weight = torch.randn(256, 128, device='cuda', dtype=torch.float16)*.02
    packed, state = bnbf.quantize_4bit(weight, blocksize=64,
                                     compress_statistics=nested, quant_type='nf4')
    x = torch.randn(3, 17, 128, device='cuda', dtype=torch.float16, requires_grad=True)
    bias = torch.randn(256, device='cuda', dtype=torch.float16)*.02
    grad = torch.randn(3, 17, 256, device='cuda', dtype=torch.float16)
    expected = torch.nn.functional.linear(x, _reference()(packed, state, x.dtype), bias)
    actual = torch.nn.functional.linear(x, dequantize_pairs(packed, state, x.dtype), bias)
    eg, = torch.autograd.grad(expected, x, grad)
    ag, = torch.autograd.grad(actual, x, grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ag, eg, rtol=0, atol=0)


def test_paired_warmup_compiles():
    from solution.paired_nf4 import warmup
    warmup()

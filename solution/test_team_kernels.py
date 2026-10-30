"""CUDA checks for our reductions, gradients, masking and Qwen cast semantics."""

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('scale', [0.5, 1.0, 2.0])
def test_lora_scale_add_exact_forward_backward(dtype, scale):
    from solution.lora_add import scale_add
    torch.manual_seed(41)
    base = torch.randn(3, 1031, device='cuda', requires_grad=True)
    update = torch.randn(3, 1031, device='cuda', dtype=dtype, requires_grad=True)
    expected = base + update * scale
    actual = scale_add(base, update, scale)
    grad = torch.randn_like(expected)
    eg = torch.autograd.grad(expected, (base, update), grad)
    ag = torch.autograd.grad(actual, (base, update), grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(ag, eg):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("vocab", [1031, 8192, 151936])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_vocabulary_loss_and_gradient(vocab, reduction):
    from solution.vocab import token_cross_entropy

    torch.manual_seed(19)
    x = torch.randn((7, vocab), device="cuda", requires_grad=True)
    labels = torch.randint(vocab, (7,), device="cuda")
    labels[1] = -100
    reference = F.cross_entropy(x, labels, reduction=reduction)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    reference_gradient = x.grad.clone()
    x.grad = None
    actual = token_cross_entropy(x, labels, reduction=reduction)
    actual.backward(upstream)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(x.grad, reference_gradient, rtol=5e-5, atol=2e-7)


def test_ignored_and_strided_rows():
    from solution.vocab import token_cross_entropy

    x = torch.randn((8, 1031), device="cuda")[::2].requires_grad_()
    labels = torch.full((4,), -100, device="cuda", dtype=torch.long)
    for reduction in ("none", "sum", "mean"):
        actual = token_cross_entropy(x, labels, reduction=reduction)
        expected = F.cross_entropy(x, labels, reduction=reduction)
        torch.testing.assert_close(actual, expected, equal_nan=True)
    token_cross_entropy(x, labels, reduction="sum").backward()
    assert torch.count_nonzero(x.grad).item() == 0


@pytest.mark.parametrize("vocab,scale", [(1031, 1.0), (151936, 1.0), (4099, 30.0)])
def test_entropy(vocab, scale):
    from solution.vocab import token_entropy

    torch.manual_seed(23)
    x = torch.randn((2, 3, vocab), device="cuda") * scale
    log_probs = x.log_softmax(-1)
    expected = -(log_probs.exp() * log_probs).sum(-1)
    with torch.no_grad():
        actual = token_entropy(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [1536, 1001])
def test_frozen_norm_matches_qwen_forward_backward(dtype, width):
    from solution.rmsnorm import frozen_rmsnorm

    torch.manual_seed(29)
    x = torch.randn((2, 7, width), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.float32)
    value = x.float()
    ref = weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    upstream = torch.randn_like(ref)
    ref.backward(upstream)
    expected_grad = x.grad.clone()
    x.grad = None
    actual = frozen_rmsnorm(x, weight)
    actual.backward(upstream)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, ref, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x.grad, expected_grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape,blocksize", [((127, 65), 64), ((128, 256), 256)])
def test_nf4_decoder_matches_bitsandbytes(nested, dtype, shape, blocksize):
    import bitsandbytes.functional as bnb
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(31)
    weight = torch.randn(shape, device="cuda", dtype=dtype)
    packed, state = bnb.quantize_4bit(
        weight, blocksize=blocksize, compress_statistics=nested, quant_type="nf4",
    )
    expected = bnb.dequantize_4bit(packed, state)
    actual = dequantize_nf4(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    output = torch.empty_like(expected)
    actual_out = dequantize_nf4(packed, state, output)
    assert actual_out.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual_out, expected, rtol=0, atol=0)
    transposed = dequantize_nf4(packed.t(), state)
    torch.testing.assert_close(transposed, expected.t(), rtol=0, atol=0)
    if dtype == torch.float16:
        wide_output = torch.empty(state.shape, dtype=torch.float32, device="cuda")
        widened = dequantize_nf4(packed, state, wide_output)
        torch.testing.assert_close(widened, expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("autocast", [False, True])
def test_nf4_forward_fallback_matches_backend(nested, dtype, autocast):
    import bitsandbytes.functional as bnb
    from bitsandbytes.backends.cuda import ops
    from solution.nf4 import nf4_linear_fallback

    torch.manual_seed(37)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    packed, state = bnb.quantize_4bit(weight, compress_statistics=nested, quant_type="nf4")
    x = torch.randn((2, 7, 64), device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(128, device="cuda", dtype=dtype)
    args = (x, packed, state.shape, state.state2.absmax if nested else state.absmax,
            state.blocksize, "nf4")
    kwargs = {"bias": bias}
    if nested:
        kwargs.update(absmax_8bit=state.absmax, absmax_code=state.state2.code,
                      absmax_offset=state.offset)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        expected = ops._dequant_linear_fallback(*args, **kwargs)
        actual = nf4_linear_fallback(*args, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected_grad, = torch.autograd.grad(expected, x, upstream)
    actual_grad, = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

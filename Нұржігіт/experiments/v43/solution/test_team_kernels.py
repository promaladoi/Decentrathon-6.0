"""CUDA checks for our reductions, gradients, masking and Qwen cast semantics."""

import pytest
import torch


@pytest.mark.parametrize('provided_shift', [False, True])
@pytest.mark.parametrize('count', [None, 7])
def test_half_causal_loss_matches_float_reference(provided_shift, count):
    from solution.causal_loss import causal_loss
    torch.manual_seed(43)
    x = torch.randn(2, 5, 1031, device='cuda', dtype=torch.float16, requires_grad=True)
    labels = torch.randint(1031, (2, 5), device='cuda')
    labels[:, 0] = -100
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:].contiguous()
    expected = torch.nn.functional.cross_entropy(x.float().view(-1, 1031), shifted.view(-1),
                reduction='sum' if count is not None else 'mean')
    if count is not None:
        expected = expected / count
    actual = causal_loss(x, labels, 1031, count, shift_labels=shifted if provided_shift else None)
    eg, = torch.autograd.grad(expected, x)
    ag, = torch.autograd.grad(actual, x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(ag, eg, rtol=2e-3, atol=2e-6)


import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("vocab", [1031, 8192, 151936])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vocabulary_loss_and_gradient(vocab, reduction, dtype):
    from solution.vocab import token_cross_entropy

    torch.manual_seed(19)
    x = torch.randn((7, vocab), device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(vocab, (7,), device="cuda")
    labels[1] = -100
    reference = F.cross_entropy(x.float(), labels, reduction=reduction)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    reference_gradient = x.grad.clone()
    x.grad = None
    actual = token_cross_entropy(x, labels, reduction=reduction)
    actual.backward(upstream)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(x.grad, reference_gradient,
                               rtol=2e-3 if dtype == torch.float16 else 5e-5,
                               atol=2e-6 if dtype == torch.float16 else 2e-7)


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
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_entropy(vocab, scale, dtype):
    from solution.vocab import token_entropy

    torch.manual_seed(23)
    x = (torch.randn((2, 3, vocab), device="cuda") * scale).to(dtype)
    log_probs = x.float().log_softmax(-1)
    expected = -(log_probs.exp() * log_probs).sum(-1)
    with torch.no_grad():
        actual = token_entropy(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-5)


def test_output_precision_preserves_values_fields_and_loss_gradient():
    from accelerate.utils.operations import convert_to_fp32
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from solution.output_precision import convert_outputs

    logits = torch.randn(2, 5, 1031, device='cuda', dtype=torch.float16, requires_grad=True)
    loss = logits.float().square().mean()
    hidden = torch.randn(2, 5, 16, device='cuda', dtype=torch.float16)
    source = CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=(hidden,))
    expected = convert_to_fp32(source)
    actual = convert_outputs(source, convert_to_fp32)
    assert type(actual) is type(source)
    assert list(actual.keys()) == list(expected.keys())
    assert actual.logits is logits and source.logits is logits
    torch.testing.assert_close(actual.logits.float(), expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.hidden_states[0], expected.hidden_states[0], rtol=0, atol=0)
    expected_grad, = torch.autograd.grad(expected.loss, logits, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual.loss, logits)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    plain = {'value': hidden}
    fallback = convert_outputs(plain, convert_to_fp32)
    torch.testing.assert_close(fallback['value'], hidden.float(), rtol=0, atol=0)


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

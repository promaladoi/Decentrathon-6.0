"""QKV keeps separate GEMM rounding and the original ordered gradient join."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


def _same_bits(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual[finite].view(integer), expected[finite].view(integer))


def _layers():
    layers = []
    for output in (64, 16, 16):
        weight = torch.randn((output, 65), device="cuda", dtype=torch.float16) * 0.05
        weight.quant_state = None
        layers.append(SimpleNamespace(weight=weight, bias=None))
    return tuple(layers)


@pytest.mark.parametrize("active", [(True, True, True), (True, False, True),
                                      (False, True, False), (False, False, False)])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_qkv_backward_optional_branches_match_separate_native_gemms(monkeypatch, active, input_dtype):
    import solution.unpad as unpad

    torch.manual_seed(824)
    layers = _layers()
    monkeypatch.setattr(unpad, "_weight", lambda layer, dtype: layer.weight.to(dtype))
    grads = tuple(torch.randn((17, layer.weight.shape[0]), device="cuda", dtype=torch.float16) * 128
                  if used else None for layer, used in zip(layers, active))
    ctx = SimpleNamespace(layers=layers, dtype=torch.float16, x_dtype=input_dtype)
    expected = None
    with torch.no_grad():
        for layer, grad in zip(layers, grads):
            if grad is not None:
                part = torch.matmul(grad, layer.weight).float()
                expected = part if expected is None else expected.add_(part)
        actual = unpad.QKV4bit.backward(ctx, *grads)[0]
    if expected is None:
        assert actual is None
    else:
        _same_bits(actual, expected.to(input_dtype))


def test_qkv_create_graph_preserves_upstream_second_derivatives(monkeypatch):
    import solution.unpad as unpad

    torch.manual_seed(825)
    layers = _layers()
    monkeypatch.setattr(unpad, "_weight", lambda layer, dtype: layer.weight.to(dtype))
    monkeypatch.setattr(unpad, "_linear_forward",
                        lambda x, weight, bias, state: F.linear(x, weight, bias))
    x = torch.randn((17, 65), device="cuda", requires_grad=True)
    upstream = tuple(torch.randn((17, layer.weight.shape[0]), device="cuda",
                                 dtype=torch.float16, requires_grad=True) for layer in layers)
    outputs = unpad.QKV4bit.apply(x, *layers, torch.float16)
    actual, = torch.autograd.grad(outputs, x, upstream, create_graph=True)
    assert actual.requires_grad
    expected = None
    for layer, grad in zip(layers, upstream):
        part = torch.matmul(grad, layer.weight).float()
        expected = part if expected is None else expected + part
    _same_bits(actual, expected)
    probe = torch.randn_like(x)
    actual_second = torch.autograd.grad(actual, upstream, probe)
    expected_second = torch.autograd.grad(expected, upstream, probe)
    for value, reference in zip(actual_second, expected_second):
        _same_bits(value, reference)

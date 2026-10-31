"""GPU qualification of the conservative same-team integration changes.

These tests add no benchmark claims and never modify the fixed task/harness.
Vladimir's integration and Merey's exact decoder remain credited in the modules.
"""

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _qwen_reference(x, weight, eps):
    # Qwen casts once and reuses that tensor on both normalization branches.
    # Repeating x.float() would round their gradients separately for FP16 x.
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def test_single_cast_reference_matches_pinned_qwen2_on_cpu():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(575)
    shape = (2, 17, 64)
    residual = torch.ones(shape, device='cpu', dtype=torch.float16, requires_grad=True)
    delta = torch.zeros(shape, device='cpu', dtype=torch.float16, requires_grad=True)
    weight = torch.randn(64, device='cpu', dtype=torch.float32)
    native = Qwen2RMSNorm(64, eps=0.0).cpu().requires_grad_(False)
    native.weight.copy_(weight)
    h = residual + delta
    expected_y = native(h)
    reference_y = _qwen_reference(h, weight, 0.0)
    gh, gy = torch.randn_like(h), torch.randn_like(expected_y)
    expected = torch.autograd.grad((h, expected_y), (residual, delta), (gh, gy), retain_graph=True)
    reference = torch.autograd.grad((h, reference_y), (residual, delta), (gh, gy), retain_graph=True)
    torch.testing.assert_close(reference_y, expected_y, rtol=0, atol=0)
    for value, original in zip(reference, expected):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    # Lock in the cause of the failed smoke test: this superficially similar
    # expression has TWO CastBackward nodes and is not the pinned Qwen graph.
    double_cast = weight * (h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True))).to(h.dtype)
    different = torch.autograd.grad((h, double_cast), (residual, delta), (gh, gy))
    assert any(bool(torch.any(a != b)) for a, b in zip(different, expected))


@CUDA
@pytest.mark.parametrize('weight_dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('output_dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('nested', [False, True])
def test_nf4_decode_preserves_quant_state_rounding_exactly(weight_dtype, output_dtype, nested):
    import bitsandbytes.functional as functional
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(571)
    weights = torch.randn((37, 65), device='cuda', dtype=weight_dtype)
    packed, state = functional.quantize_4bit(
        weights, blocksize=64, quant_type='nf4', compress_statistics=nested)
    reference = functional.dequantize_4bit(packed, state).to(output_dtype)
    actual = dequantize_nf4(packed, state, output_dtype)
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_nf4(packed.t(), state, output_dtype),
                               reference, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize('rows,dtype', [(2, torch.float16), (2048, torch.float16),
                                       (16, torch.float32)])
def test_nf4_forward_preserves_native_dispatch_and_exact_input_gradient(monkeypatch, rows, dtype):
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import matmul_4bit as native_matmul
    from solution import nf4

    torch.manual_seed(572)
    weights = torch.randn((256, 1536), device='cuda', dtype=torch.float16) * 0.02
    packed, state = functional.quantize_4bit(weights, quant_type='nf4', compress_statistics=True)
    x = torch.randn((rows, 1536), device='cuda', dtype=dtype, requires_grad=True)
    upstream = torch.randn((rows, 256), device='cuda', dtype=dtype)
    expected = native_matmul(x, packed, state)
    expected_dx, = torch.autograd.grad(expected, (x,), upstream)
    decodes = []
    original_decode = nf4._decode
    def decode(*args):
        decodes.append(args[2])
        return original_decode(*args)
    monkeypatch.setattr(nf4, '_decode', decode)
    native_custom = nf4._uses_native_custom_gemm(x, state)
    actual = nf4.MatMulNF4.apply(x, packed, None, state)
    assert len(decodes) == (0 if native_custom else 1)
    actual_dx, = torch.autograd.grad(actual, (x,), upstream)
    assert len(decodes) == (1 if native_custom else 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)


def _quantized_mlp():
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    config = Qwen2Config(hidden_size=64, intermediate_size=192, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1, hidden_act='silu')
    mlp = Qwen2MLP(config)
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        old = getattr(mlp, name)
        layer = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=False,
                                 compute_dtype=torch.float16, quant_type='nf4',
                                 compress_statistics=True)
        layer.weight.data.copy_(old.weight)
        setattr(mlp, name, layer.to(torch.float16).cuda())
    return mlp


@CUDA
@pytest.mark.parametrize('rows', [31, 604])
def test_public_mlp_dtype_and_retained_graph_match_native_projection_path(rows):
    from solution.mlp import _compute_dtype, make_mlp_forward
    from solution.swiglu import swiglu

    torch.manual_seed(573)
    mlp = _quantized_mlp()
    assert _compute_dtype((mlp.gate_proj, mlp.up_proj, mlp.down_proj)) == torch.float16
    x = torch.randn((rows, 64), device='cuda', dtype=torch.float32, requires_grad=True)
    upstream = torch.randn_like(x) * 64.0
    def previous(module, values):
        return module.down_proj(swiglu(module.gate_proj(values), module.up_proj(values)))
    forward = make_mlp_forward(previous)
    with torch.autocast('cuda', dtype=torch.float16):
        expected = previous(mlp, x)
        actual = forward(mlp, x)
    expected_dx, = torch.autograd.grad(expected, (x,), upstream)
    first_dx, = torch.autograd.grad(actual, (x,), upstream, retain_graph=True)
    second_dx, = torch.autograd.grad(actual, (x,), upstream)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(first_dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(second_dx, first_dx, rtol=0, atol=0)


@CUDA
def test_mlp_guard_keeps_observed_projection_on_previous_forward():
    from solution.mlp import _compute_dtype, make_mlp_forward

    mlp = _quantized_mlp()
    handle = mlp.gate_proj.register_forward_hook(lambda module, args, result: result)
    marker = object()
    try:
        assert _compute_dtype((mlp.gate_proj, mlp.up_proj, mlp.down_proj)) is None
        forward = make_mlp_forward(lambda module, x: marker)
        assert forward(mlp, torch.ones((2, 64), device='cuda')) is marker
    finally:
        handle.remove()


@CUDA
def test_half_rmsnorm_backward_keeps_native_intermediate_gradient_rounding():
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(574)
    x = torch.ones((3, 64), device='cuda', dtype=torch.float16, requires_grad=True)
    weight = torch.randn(64, device='cuda', dtype=torch.float32)
    upstream = torch.randn((3, 64), device='cuda', dtype=torch.float32)
    xf = x.float()
    reference = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True))).to(x.dtype)
    expected_dx, = torch.autograd.grad(reference, (x,), upstream)
    actual = rmsnorm(x, weight, 0.0)
    actual_dx, = torch.autograd.grad(actual, (x,), upstream)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize('residual_dtype,delta_dtype', [
    (torch.float16, torch.float16), (torch.float32, torch.float16),
    (torch.float32, torch.float32)])
def test_add_rmsnorm_preserves_both_gradient_branches(residual_dtype, delta_dtype):
    from solution.rmsnorm import add_rmsnorm

    torch.manual_seed(575)
    shape = (2, 17, 64)
    residual = torch.ones(shape, device='cuda', dtype=residual_dtype, requires_grad=True)
    delta = torch.zeros(shape, device='cuda', dtype=delta_dtype, requires_grad=True)
    weight = torch.randn(64, device='cuda', dtype=torch.float32)
    h = residual + delta
    y = _qwen_reference(h, weight, 0.0)
    gh, gy = torch.randn_like(h), torch.randn_like(y)
    expected = torch.autograd.grad((h, y), (residual, delta), (gh, gy))
    actual_h, actual_y = add_rmsnorm(residual, delta, weight, 0.0)
    actual = torch.autograd.grad((actual_h, actual_y), (residual, delta), (gh, gy))
    torch.testing.assert_close(actual_h, h, rtol=0, atol=0)
    torch.testing.assert_close(actual_y, y, rtol=0, atol=0)
    # All-half inputs exercise the exact CastBackward→branch-join boundary.
    tolerance = 0.0 if residual_dtype == delta_dtype == torch.float16 else 1e-5
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)

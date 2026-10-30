"""Check the cast-sensitive Qwen norm against its eager formula."""

import pytest
import torch


def _reference(x, weight, eps):
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 17, 1001), (7, 1)])
@pytest.mark.parametrize("input_dtype,weight_dtype", [
    (torch.float16, torch.float32), (torch.float32, torch.float32),
    (torch.float16, torch.float16),
])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_rmsnorm_forward_and_backward(shape, input_dtype, weight_dtype, eps):
    from solution.rmsnorm import frozen_rmsnorm

    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=input_dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype)
    out = frozen_rmsnorm(x, weight, eps)
    ref = _reference(ref_x, weight, eps)
    grad = torch.randn_like(ref)
    out.backward(grad)
    ref.backward(grad)
    assert out.dtype == ref.dtype
    # Half rounding around a midpoint may differ by one ulp; its backward
    # projection is also sensitive to that intermediate cast.
    tol = 3e-3 if input_dtype == torch.float16 else 3e-5
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_rmsnorm_handles_zero_and_small_values():
    from solution.rmsnorm import frozen_rmsnorm

    for magnitude in (0.0, 1e-5):
        x = torch.full((5, 1536), magnitude, device="cuda", dtype=torch.float16,
                       requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        weight = torch.ones(1536, device="cuda", dtype=torch.float32)
        out, ref = frozen_rmsnorm(x, weight), _reference(ref_x, weight, 1e-6)
        out.sum().backward()
        ref.sum().backward()
        torch.testing.assert_close(out, ref, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_rmsnorm_falls_back_when_weight_is_trainable(noncontiguous):
    from solution.rmsnorm import frozen_rmsnorm

    x = torch.randn((3, 7, 11), device="cuda", requires_grad=True)
    if noncontiguous:
        x = x.transpose(0, 1).detach().requires_grad_(True)
    weight = torch.randn(11, device="cuda", requires_grad=True)
    out, ref = frozen_rmsnorm(x, weight), _reference(x, weight, 1e-6)
    actual = torch.autograd.grad(out.sum(), (x, weight), retain_graph=True)
    expected = torch.autograd.grad(ref.sum(), (x, weight))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_rmsnorm_noncontiguous_frozen_input_uses_eager():
    from solution.rmsnorm import frozen_rmsnorm

    x = torch.randn((2, 5, 13), device="cuda").transpose(0, 1).requires_grad_(True)
    weight = torch.randn(13, device="cuda")
    out, ref = frozen_rmsnorm(x, weight), _reference(x, weight, 1e-6)
    actual = torch.autograd.grad(out.sum(), x, retain_graph=True)[0]
    expected = torch.autograd.grad(ref.sum(), x)[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_qwen_rmsnorm_patch_is_idempotent_and_cpu_falls_back():
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    patch_qwen2_rmsnorm()
    first = Qwen2RMSNorm.forward
    patch_qwen2_rmsnorm()
    assert Qwen2RMSNorm.forward is first
    norm = Qwen2RMSNorm(13)
    x = torch.randn((3, 13), requires_grad=True)
    torch.testing.assert_close(norm(x), _reference(x, norm.weight, norm.variance_epsilon),
                               rtol=0, atol=0)

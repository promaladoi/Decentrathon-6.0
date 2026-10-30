"""Qwen RoPE correctness, casts, transpose views and frozen-phase fallbacks."""

import inspect

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires CUDA and Triton runtime")


def _module():
    pytest.importorskip("triton")
    from solution import rope

    return rope


def _reference(q, k, cos, sin, unsqueeze_dim=1):
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    halfq, halfk = q.shape[-1] // 2, k.shape[-1] // 2
    rq = torch.cat((-q[..., halfq:], q[..., :halfq]), dim=-1)
    rk = torch.cat((-k[..., halfk:], k[..., :halfk]), dim=-1)
    return q * c + rq * s, k * c + rk * s


def _input(batch, heads, seq, width, dtype, transposed, requires_grad=True):
    shape = (batch, seq, heads, width) if transposed else (batch, heads, seq, width)
    value = torch.randn(shape, device="cuda", dtype=dtype)
    if transposed:
        value = value.transpose(1, 2)
    return value.detach().requires_grad_(requires_grad)


@pytest.mark.parametrize("shape", [(2, 3, 1, 7, 16), (1, 12, 2, 11, 128),
                                   (2, 5, 2, 3, 32)])
@pytest.mark.parametrize("query_dtype,phase_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.float32, torch.float16), (torch.float32, torch.float32)])
@pytest.mark.parametrize("broadcast_phase", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_rotary_forward_backward_preserves_casts_and_layout(
        shape, query_dtype, phase_dtype, broadcast_phase, transposed):
    rope = _module()
    batch, hq, hk, seq, width = shape
    torch.manual_seed(111)
    q = _input(batch, hq, seq, width, query_dtype, transposed)
    k = _input(batch, hk, seq, width, query_dtype, transposed)
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    phase_batch = 1 if broadcast_phase else batch
    # Distinct values in the two halves exercise the full operation contract,
    # without assuming the duplicated phases of the fixed model.
    angles = torch.randn((phase_batch, seq, width), device="cuda")
    cos, sin = angles.cos().to(phase_dtype), angles.sin().to(phase_dtype)
    out_q, out_k = rope.rotary_pos_emb(q, k, cos, sin)
    expected_q, expected_k = _reference(ref_q, ref_k, cos, sin)
    assert out_q.dtype == expected_q.dtype and out_k.dtype == expected_k.dtype
    # A strided upstream gradient must not be mistaken for contiguous storage.
    dy_q = torch.randn((*out_q.shape[:-1], width * 2), device="cuda",
                       dtype=out_q.dtype)[..., ::2]
    dy_k = torch.randn((*out_k.shape[:-1], width * 2), device="cuda",
                       dtype=out_k.dtype)[..., ::2]
    actual_grad = torch.autograd.grad((out_q, out_k), (q, k), (dy_q, dy_k))
    expected_grad = torch.autograd.grad((expected_q, expected_k), (ref_q, ref_k), (dy_q, dy_k))
    forward_tol = 2e-3 if out_q.dtype == torch.float16 else 3e-5
    grad_tol = 2e-3 if query_dtype == torch.float16 else 3e-5
    torch.testing.assert_close(out_q, expected_q, rtol=forward_tol, atol=forward_tol)
    torch.testing.assert_close(out_k, expected_k, rtol=forward_tol, atol=forward_tol)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=grad_tol, atol=grad_tol)


@pytest.mark.parametrize("q_requires,k_requires", [(True, False), (False, True), (True, True)])
def test_rotary_independent_input_gradients_and_broadcast_upstream(q_requires, k_requires):
    rope = _module()
    torch.manual_seed(112)
    q = _input(2, 3, 7, 32, torch.float16, True, q_requires)
    k = _input(2, 1, 7, 32, torch.float16, True, k_requires)
    ref_q = q.detach().clone().requires_grad_(q_requires)
    ref_k = k.detach().clone().requires_grad_(k_requires)
    angles = torch.randn((1, 7, 32), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    out_q, out_k = rope.rotary_pos_emb(q, k, cos, sin)
    expected_q, expected_k = _reference(ref_q, ref_k, cos, sin)
    assert out_q.requires_grad == q_requires and out_k.requires_grad == k_requires
    actual_loss = (out_q.sum() if q_requires else 0) + (out_k.sum() if k_requires else 0)
    expected_loss = ((expected_q.sum() if q_requires else 0)
                     + (expected_k.sum() if k_requires else 0))
    actual_loss.backward()
    expected_loss.backward()
    for value, reference, needed in ((q, ref_q, q_requires), (k, ref_k, k_requires)):
        if needed:
            torch.testing.assert_close(value.grad, reference.grad, rtol=2e-3, atol=2e-3)
        else:
            assert value.grad is None


def test_rotary_unused_output_has_no_gradient_and_does_not_mutate_inputs():
    rope = _module()
    torch.manual_seed(113)
    q = _input(2, 3, 7, 32, torch.float32, True)
    k = _input(2, 1, 7, 32, torch.float32, True)
    angles = torch.randn((2, 7, 32), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    before = [value.detach().clone() for value in (q, k, cos, sin)]
    out_q, _ = rope.rotary_pos_emb(q, k, cos, sin)
    grad_q, grad_k = torch.autograd.grad(out_q.sum(), (q, k), allow_unused=True)
    assert grad_q is not None and grad_k is None
    for value, reference in zip((q, k, cos, sin), before):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width,trainable_phase", [(13, False), (13, True), (16, True)])
def test_rotary_trainable_phases_and_odd_width_use_eager(device, width, trainable_phase):
    rope = _module()
    torch.manual_seed(114)
    # An odd width is supported by the Python fallback, not by paired kernels.
    q = torch.randn((2, 3, 5, width), device=device, requires_grad=True)
    k = torch.randn((2, 1, 5, width), device=device, requires_grad=True)
    cos = torch.randn((2, 5, width), device=device, requires_grad=trainable_phase)
    sin = torch.randn((2, 5, width), device=device, requires_grad=trainable_phase)
    out = rope.rotary_pos_emb(q, k, cos, sin)
    expected = _reference(q, k, cos, sin)
    parameters = (q, k, cos, sin) if trainable_phase else (q, k)
    actual_grad = torch.autograd.grad(sum(value.sum() for value in out),
                                      parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(sum(value.sum() for value in expected), parameters)
    for actual, reference in zip((*out, *actual_grad), (*expected, *expected_grad)):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_rotary_alternate_unsqueeze_dimension_uses_eager():
    rope = _module()
    torch.manual_seed(115)
    q = torch.randn((2, 5, 3, 16), device="cuda", requires_grad=True)
    k = torch.randn((2, 5, 1, 16), device="cuda", requires_grad=True)
    cos = torch.randn((2, 5, 16), device="cuda")
    sin = torch.randn_like(cos)
    out, expected = rope.rotary_pos_emb(q, k, cos, sin, 2), _reference(q, k, cos, sin, 2)
    for actual, reference in zip(out, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_rotary_patch_is_idempotent_and_keeps_cpu_eager(monkeypatch):
    rope = _module()
    from transformers.models.qwen2 import modeling_qwen2

    original = inspect.unwrap(modeling_qwen2.apply_rotary_pos_emb)
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", original)
    rope.patch_qwen2_rope()
    first = modeling_qwen2.apply_rotary_pos_emb
    rope.patch_qwen2_rope()
    assert modeling_qwen2.apply_rotary_pos_emb is first
    torch.manual_seed(116)
    q, k = torch.randn(2, 3, 5, 16), torch.randn(2, 1, 5, 16)
    cos, sin = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    actual, expected = first(q, k, cos, sin), original(q, k, cos, sin)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)

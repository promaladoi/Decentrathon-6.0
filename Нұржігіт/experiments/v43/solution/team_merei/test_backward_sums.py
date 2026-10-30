# Team contribution by Merei; source 29419f43d9b9173d1ba5d6824a31589ad3b84df7.
"""Ordered FP16-GEMM gradient sums, exceptional values and graph fallback."""

import pytest
import torch
pytest.importorskip("triton")


def _reference(parts):
    result = parts[0].float()
    for part in parts[1:]:
        result = result + part.float()
    return result


def _exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].contiguous().view(torch.int32),
                       expected[finite].contiguous().view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("arity", [2, 3])
@pytest.mark.parametrize("shape", [(1,), (7,), (1023,), (1024,), (1025,), (7, 1536)])
def test_fused_sum_matches_ordered_torch_and_preserves_input_bits(arity, shape):
    from solution.team_merei.backward_sums import sum_half_gradients

    special = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"),
                            65504.0, -65504.0, 2 ** -24, -(2 ** -24)],
                           device="cuda", dtype=torch.float16)
    parts = []
    for index in range(arity):
        part = torch.randn(shape, device="cuda", dtype=torch.float16)
        count = min(part.numel(), special.numel())
        part.flatten()[:count] = special.roll(index)[:count]
        parts.append(part)
    before = [part.clone().view(torch.int16) for part in parts]
    expected = _reference(parts)
    state = torch.cuda.get_rng_state()
    with torch.no_grad():
        actual = sum_half_gradients(parts)
    assert actual.dtype == torch.float32 and actual.shape == parts[0].shape
    _exact(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    for part, original in zip(parts, before):
        assert torch.equal(part.view(torch.int16), original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("arity", [2, 3])
def test_casts_precede_addition_and_three_way_sum_is_left_associative(arity):
    from solution.team_merei.backward_sums import sum_half_gradients

    # Finite half operands can overflow if added before the FP32 conversion.
    # Cancellation with a half subnormal distinguishes (a+b)+c from a+(b+c).
    parts = [torch.tensor(values, device="cuda", dtype=torch.float16) for values in
             ([65504.0, 65504.0, -0.0, 0.0],
              [65504.0, -65504.0, -0.0, -0.0],
              [-65504.0, 2 ** -24, -0.0, -0.0])][:arity]
    with torch.no_grad():
        actual = sum_half_gradients(parts)
    _exact(actual, _reference(parts))
    assert actual[0].item() == (131008.0 if arity == 2 else 65504.0)
    if arity == 3:
        assert actual[1].item() == 2 ** -24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("arity", [2, 3])
@pytest.mark.parametrize("n", [1, 7, 1023, 1025])
def test_sum_kernel_tail_never_writes_output_guards(arity, n):
    from solution.team_merei.backward_sums import BLOCK, sum_half_gradients_kernel
    import triton

    parts = [torch.randn(n, device="cuda", dtype=torch.float16) for _ in range(arity)]
    output = torch.full((n + 2,), 923.0, device="cuda")
    sum_half_gradients_kernel[(triton.cdiv(n, BLOCK),)](
        parts[0], parts[1], parts[-1], output[1:-1], n,
        THREE=arity == 3, BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)
    _exact(output[1:-1], _reference(parts))
    assert torch.equal(output[[0, -1]], torch.tensor([923.0, 923.0], device="cuda"))


class _ForbiddenKernel:
    def __getitem__(self, _grid):
        raise AssertionError("torch fallback must not launch the raw Triton kernel")


@pytest.mark.parametrize("arity", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_fallback_noncontiguous_cpu_and_dtype_preserves_values_and_inputs(monkeypatch, arity, dtype):
    import solution.team_merei.backward_sums as sums

    monkeypatch.setattr(sums, "sum_half_gradients_kernel", _ForbiddenKernel())
    parts = [torch.randn((9, 5), dtype=dtype).t() for _ in range(arity)]
    assert all(not part.is_contiguous() for part in parts)
    before = [part.clone() for part in parts]
    with torch.no_grad():
        actual = sums.sum_half_gradients(parts)
    _exact(actual, _reference(parts))
    for part, original in zip(parts, before):
        torch.testing.assert_close(part, original, rtol=0, atol=0)


@pytest.mark.parametrize("arity", [2, 3])
def test_double_backward_uses_ordered_torch_graph(monkeypatch, arity):
    import solution.team_merei.backward_sums as sums

    monkeypatch.setattr(sums, "sum_half_gradients_kernel", _ForbiddenKernel())
    parts = [torch.randn((3, 7), dtype=torch.float16, requires_grad=True) for _ in range(arity)]
    reference = [part.detach().clone().requires_grad_(True) for part in parts]
    actual = sums.sum_half_gradients(parts)
    expected = _reference(reference)
    first = torch.autograd.grad(actual.square().sum(), parts, create_graph=True)
    expected_first = torch.autograd.grad(expected.square().sum(), reference, create_graph=True)
    second = torch.autograd.grad(sum(value.float().sum() for value in first), parts)
    expected_second = torch.autograd.grad(sum(value.float().sum() for value in expected_first), reference)
    _exact(actual, expected)
    for actual_grad, expected_grad in zip((*first, *second), (*expected_first, *expected_second)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_enabled_grad_mode_and_requires_grad_inputs_keep_torch_fallback(monkeypatch):
    import solution.team_merei.backward_sums as sums

    monkeypatch.setattr(sums, "sum_half_gradients_kernel", _ForbiddenKernel())
    parts = [torch.randn(17, device="cuda", dtype=torch.float16) for _ in range(2)]
    _exact(sums.sum_half_gradients(parts), _reference(parts))
    parts[0].requires_grad_()
    with torch.no_grad():
        _exact(sums.sum_half_gradients(parts), _reference(parts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_compile_only_warmup_keeps_rng():
    from solution.team_merei.backward_sums import warmup

    before = torch.cuda.get_rng_state()
    warmup()
    assert torch.equal(before, torch.cuda.get_rng_state())

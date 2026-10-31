"""Actual-scale backward parity for the team's chunked frozen lm_head.

The reference is ordinary half Linear -> explicit float CE -> autograd, not
the fixed-scale teammate implementation. Chunked cuBLAS may choose a different
reduction schedule: finite hidden gradients use relative L2 <= 0.2%; ignored
rows and nonfinite categories must match exactly. The two-class rounding
witness below additionally checks the actual scale placement exactly.
Saved-state checks require the original supervised FP16 logits and FP32 LSE,
with no vocabulary-sized cache copy or repeated head-forward GEMM in backward.
The strict private GradScaler path is also checked with matching and differing
actual upstream values: only a matching value may reuse precomputed gradients.
"""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


def _native(x, weight, targets, denominator):
    logits = F.linear(x.to(torch.float16), weight.to(torch.float16)).float()
    if denominator is None:
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
    else:
        loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
        if torch.is_tensor(denominator):
            denominator = denominator.to(loss.device)
        loss = loss / denominator
    return loss


def _assert_gradient(actual, expected, ignored):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual[ignored], expected[ignored], rtol=0, atol=0)
    finite = torch.isfinite(expected)
    if finite.any():
        difference = (actual[finite].float() - expected[finite].float()).norm()
        reference = expected[finite].float().norm().clamp(min=1e-12)
        assert (difference / reference).item() <= 2e-3


@pytest.mark.parametrize("upstream", [1.0, 1024.0, 65536.0])
@pytest.mark.parametrize("normalizer", ["mean", "integer", "cpu_tensor", "gpu_tensor", "negative"])
def test_actual_scale_matches_native_half_rounding(upstream, normalizer):
    from solution.lmhead import lm_head_loss

    torch.manual_seed(321)
    tokens, hidden, vocabulary = 23, 32, 4103
    x = torch.randn((tokens, hidden), device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    weight = torch.randn((vocabulary, hidden), device="cuda") * 0.05
    targets = torch.randint(vocabulary, (tokens,), device="cuda")
    targets[::3] = -100
    targets[-1] = vocabulary - 1
    denominator = {"mean": None, "integer": 31, "cpu_tensor": torch.tensor(31),
                   "gpu_tensor": torch.tensor(31, device="cuda"), "negative": -17}[normalizer]
    expected = _native(ref_x, weight, targets, denominator)
    actual, stats = lm_head_loss(x, weight, targets, denominator)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.backward(torch.tensor(upstream, device="cuda"))
    expected.backward(torch.tensor(upstream, device="cuda"))
    _assert_gradient(x.grad, ref_x.grad, targets == -100)
    assert torch.count_nonzero(stats["ce"][targets == -100]).item() == 0
    assert torch.equal(stats["valid"], targets != -100)


@pytest.mark.parametrize("upstream", [1.0, 1024.0, 65536.0])
def test_multiple_supervised_chunks_match_native_gradient(upstream):
    from solution.lmhead import lm_head_loss, CHUNK

    torch.manual_seed(322)
    tokens, hidden, vocabulary = CHUNK + 189, 128, 10007
    x = torch.randn((tokens, hidden), device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    weight = torch.randn((vocabulary, hidden), device="cuda") * 0.05
    targets = torch.randint(vocabulary, (tokens,), device="cuda")
    targets[::7] = -100
    assert (targets != -100).sum().item() > CHUNK
    denominator = (targets != -100).sum()
    expected = _native(ref_x, weight, targets, denominator)
    actual, _ = lm_head_loss(x, weight, targets, denominator)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.backward(torch.tensor(upstream, device="cuda"))
    expected.backward(torch.tensor(upstream, device="cuda"))
    _assert_gradient(x.grad, ref_x.grad, targets == -100)


def test_two_class_scale_placement_rounding_witness():
    from solution.lmhead import lm_head_loss

    x = torch.zeros((1, 1), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.tensor([[0.1234130859375], [-0.0740966796875]], device="cuda")
    targets = torch.tensor([0], device="cuda")
    expected = _native(reference, weight, targets, 3)
    actual, _ = lm_head_loss(x, weight, targets, 3)
    expected.backward(torch.tensor(65536.0, device="cuda"))
    actual.backward(torch.tensor(65536.0, device="cuda"))
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
    assert reference.grad.item() == -2156.0


@pytest.mark.parametrize("all_ignored", [False, True])
@pytest.mark.parametrize("normalizer", ["mean", "zero"])
@pytest.mark.parametrize("upstream", [1.0, float("inf"), -float("inf"), float("nan")])
def test_nonfinite_and_empty_supervision_matches_native(all_ignored, normalizer, upstream):
    from solution.lmhead import lm_head_loss

    torch.manual_seed(323)
    x = torch.randn((9, 16), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.full((9,), -100, device="cuda", dtype=torch.int64)
    if not all_ignored:
        targets[1::2] = torch.randint(97, (4,), device="cuda")
    denominator = None if normalizer == "mean" else torch.tensor(0)
    expected = _native(reference, weight, targets, denominator)
    actual, _ = lm_head_loss(x, weight, targets, denominator)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True)
    expected.backward(torch.tensor(upstream, device="cuda"))
    actual.backward(torch.tensor(upstream, device="cuda"))
    _assert_gradient(x.grad, reference.grad, targets == -100)
    if all_ignored:
        assert torch.count_nonzero(x.grad).item() == 0


def test_bounds_flag_matches_native_domain_without_device_assert():
    from solution.lmhead import _targets_in_bounds

    assert _targets_in_bounds(torch.tensor([-100, 0, 96], device="cuda"), 97).item()
    for invalid in (-101, -1, 97, 1000):
        values = torch.tensor([0, invalid], device="cuda", dtype=torch.int64)
        assert not _targets_in_bounds(values, 97).item()
        # Test the native invalid-label exception on CPU to leave this GPU
        # process usable; production uses the same checked flag asynchronously.
        with pytest.raises((IndexError, RuntimeError)):
            F.cross_entropy(torch.zeros((2, 97)), values.cpu())


def test_trainable_head_is_refused_before_losing_its_gradient():
    from solution.lmhead import lm_head_loss

    x = torch.randn((3, 16), device="cuda", requires_grad=True)
    weight = torch.randn((97, 16), device="cuda", requires_grad=True)
    targets = torch.ones(3, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="frozen weight"):
        lm_head_loss(x, weight, targets, None)


def test_saved_state_retains_original_labelled_half_logits_and_lse(monkeypatch):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "CHUNK", 3)
    original_mm = torch.mm
    calls = []

    def record_mm(left, right):
        result = original_mm(left, right)
        calls.append((tuple(left.shape), tuple(right.shape), result))
        return result

    monkeypatch.setattr(torch, "mm", record_mm)
    x = torch.randn((9, 16), device="cuda", requires_grad=True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.tensor([-100, 0, -100, 3, -100, 7, -100, 96, -100], device="cuda")
    loss, _ = lmhead.lm_head_loss(x, weight, targets, None)
    half_weight, indices, labelled_targets, denominator, *buffers = loss.grad_fn.saved_tensors
    assert half_weight.dtype == torch.float16 and half_weight.shape == weight.shape
    assert torch.equal(indices, (targets != -100).nonzero().squeeze(1))
    assert torch.equal(labelled_targets, targets.index_select(0, indices))
    assert denominator.item() == 4
    assert loss.grad_fn.num_chunks == 2
    logits_chunks, lse_chunks = buffers[:2], buffers[2:]
    assert [tuple(value.shape) for value in logits_chunks] == [(3, 97), (1, 97)]
    assert [tuple(value.shape) for value in lse_chunks] == [(3,), (1,)]
    for position, (logits, lse) in enumerate(zip(logits_chunks, lse_chunks)):
        assert logits.dtype == torch.float16
        assert lse.dtype == torch.float32
        # Labelled rows run first; retaining that original GEMM output avoids a
        # second half cache allocation. Prompt-row GEMM outputs are not saved.
        assert logits.data_ptr() == calls[position][2].data_ptr()
        torch.testing.assert_close(lse, logits.float().logsumexp(dim=-1), rtol=1e-5, atol=1e-5)
    forward_calls = len(calls)
    assert forward_calls == 4  # two labelled chunks and two prompt chunks
    loss.backward()
    assert len(calls) == forward_calls + 2
    assert all(right == (97, 16) for _, right, _ in calls[forward_calls:])
    assert torch.count_nonzero(x.grad[targets == -100]).item() == 0


@pytest.mark.parametrize("expected_scale", [1.0, 1024.0, 65536.0])
@pytest.mark.parametrize("actual_scale", [1.0, 1024.0, 65536.0])
@pytest.mark.parametrize("normalizer", ["mean", "integer"])
def test_private_expected_scale_and_mismatch_match_native(
        monkeypatch, expected_scale, actual_scale, normalizer):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: expected_scale)
    monkeypatch.setattr(lmhead, "CHUNK", 7)
    torch.manual_seed(324)
    x = torch.randn((23, 32), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((4103, 32), device="cuda") * 0.05
    targets = torch.randint(4103, (23,), device="cuda")
    targets[::3] = -100
    denominator = None if normalizer == "mean" else 31
    expected = _native(reference, weight, targets, denominator)
    actual, _ = lmhead.lm_head_loss(x, weight, targets, denominator)
    assert actual.grad_fn.precomputed
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    expected.backward(torch.tensor(actual_scale, device="cuda"))
    actual.backward(torch.tensor(actual_scale, device="cuda"))
    _assert_gradient(x.grad, reference.grad, targets == -100)


@pytest.mark.parametrize("actual_scale", [1.0, 65536.0])
def test_private_scale_placement_exact_rounding_witness(monkeypatch, actual_scale):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 65536.0)
    x = torch.zeros((1, 1), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.tensor([[0.1234130859375], [-0.0740966796875]], device="cuda")
    targets = torch.tensor([0], device="cuda")
    expected = _native(reference, weight, targets, 3)
    actual, _ = lmhead.lm_head_loss(x, weight, targets, 3)
    expected.backward(torch.tensor(actual_scale, device="cuda"))
    actual.backward(torch.tensor(actual_scale, device="cuda"))
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("actual_scale", [1024.0, 1.0])
def test_private_saved_state_and_backward_gemm_schedule(monkeypatch, actual_scale):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 1024.0)
    monkeypatch.setattr(lmhead, "CHUNK", 3)
    original_mm = torch.mm
    calls = []

    def record_mm(left, right):
        result = original_mm(left, right)
        calls.append((tuple(left.shape), tuple(right.shape)))
        return result

    monkeypatch.setattr(torch, "mm", record_mm)
    x = torch.randn((9, 16), device="cuda", requires_grad=True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.tensor([-100, 0, -100, 3, -100, 7, -100, 96, -100], device="cuda")
    loss, _ = lmhead.lm_head_loss(x, weight, targets, None)
    half_weight, indices, labels, denominator, x16, scale, dx, *lses = loss.grad_fn.saved_tensors
    assert half_weight.shape == (97, 16) and half_weight.dtype == torch.float16
    assert x16.shape == x.shape and x16.dtype == torch.float16
    assert x16.data_ptr() != x.data_ptr()
    assert dx.shape == x.shape and dx.dtype == torch.float16
    assert scale.item() == 1024.0 and scale.dtype == torch.float32
    assert [tuple(value.shape) for value in lses] == [(3,), (1,)]
    assert all(value.dtype == torch.float32 for value in lses)
    assert not any(value.ndim == 2 and value.shape[-1] == 97
                   for value in loss.grad_fn.saved_tensors)
    assert torch.equal(labels, targets.index_select(0, indices))
    assert denominator.item() == 4
    forward_calls = len(calls)
    assert forward_calls == 6  # two head+DX pairs, then two prompt heads
    loss.backward(torch.tensor(actual_scale, device="cuda"))
    after = calls[forward_calls:]
    if actual_scale == 1024.0:
        assert after == []
    else:
        assert after == [((3, 16), (16, 97)), ((3, 97), (97, 16)),
                         ((1, 16), (16, 97)), ((1, 97), (97, 16))]
    assert torch.count_nonzero(x.grad[targets == -100]).item() == 0


@pytest.mark.parametrize("all_ignored", [False, True])
@pytest.mark.parametrize("normalizer", ["mean", "zero"])
@pytest.mark.parametrize("upstream", [65536.0, float("inf"), -float("inf"), float("nan")])
def test_private_nonfinite_and_ignore_semantics(monkeypatch, all_ignored, normalizer, upstream):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 65536.0)
    torch.manual_seed(325)
    x = torch.randn((9, 16), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.full((9,), -100, device="cuda", dtype=torch.int64)
    if not all_ignored:
        targets[1::2] = torch.randint(97, (4,), device="cuda")
    denominator = None if normalizer == "mean" else 0
    expected = _native(reference, weight, targets, denominator)
    actual, _ = lmhead.lm_head_loss(x, weight, targets, denominator)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True)
    expected.backward(torch.tensor(upstream, device="cuda"))
    actual.backward(torch.tensor(upstream, device="cuda"))
    _assert_gradient(x.grad, reference.grad, targets == -100)


def test_private_precompute_requires_owned_fp32_input_copy(monkeypatch):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 1024.0)
    x = torch.randn((3, 16), device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.ones(3, device="cuda", dtype=torch.int64)
    loss, _ = lmhead.lm_head_loss(x, weight, targets, None)
    assert not loss.grad_fn.precomputed
    loss.backward()


def test_private_mismatch_uses_owned_hidden_copy_after_input_mutation(monkeypatch):
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 1024.0)
    torch.manual_seed(326)
    x = torch.randn((9, 16), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((97, 16), device="cuda") * 0.05
    targets = torch.randint(97, (9,), device="cuda")
    targets[::3] = -100
    expected = _native(reference, weight, targets, None)
    actual, _ = lmhead.lm_head_loss(x, weight, targets, None)
    with torch.no_grad():
        x.add_(100.0)
    expected.backward()
    actual.backward()  # upstream 1 differs from the saved 1024
    _assert_gradient(x.grad, reference.grad, targets == -100)

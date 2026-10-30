"""Numerical tests for the tiled vocabulary kernels, independent of Soup."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires CUDA and Triton")


def _module():
    pytest.importorskip("triton")
    from solution import vocab
    return vocab


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("rows,width", [(7, 3), (5, 1001), (6, 4096),
                                         (5, 4103), (4, 8193), (3, 151936)])
def test_cross_entropy_forward_and_backward(reduction, rows, width):
    vocab = _module()
    torch.manual_seed(61)
    logits = (torch.randn((rows, width), device="cuda") * 3).requires_grad_(True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(width, (rows,), device="cuda")
    targets[1] = -100
    targets[-1] = width - 1  # Include the last, possibly partial, tile.
    actual = vocab.cross_entropy(logits, targets, reduction=reduction)
    expected = F.cross_entropy(reference_logits, targets, reduction=reduction)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=2e-5, atol=3e-7)
    assert torch.count_nonzero(logits.grad[1]) == 0


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_all_ignored_matches_pytorch(reduction):
    vocab = _module()
    torch.manual_seed(62)
    logits = torch.randn((4, 4103), device="cuda", requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.full((4,), -17, device="cuda", dtype=torch.long)
    actual = vocab.cross_entropy(logits, targets, ignore_index=-17, reduction=reduction)
    expected = F.cross_entropy(ref_logits, targets, ignore_index=-17, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if reduction == "mean":
        assert actual.isnan().item()
    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=0, atol=0)
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_wide_finite_logits_and_nondefault_ignore(reduction):
    vocab = _module()
    # A constant offset must not cause exp overflow or cancellation in CE.
    # Large ranges also exercise exp underflow in tiles away from the maximum.
    torch.manual_seed(63)
    logits = (torch.randn((5, 8201), device="cuda") * 2000 + 10000).requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 4100, -7, 8200, 1], device="cuda")
    actual = vocab.cross_entropy(logits, targets, ignore_index=-7, reduction=reduction)
    expected = F.cross_entropy(ref_logits, targets, ignore_index=-7, reduction=reduction)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-4)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("shape", [(2, 3, 1001), (3, 4103), (2, 151936), (5, 1)])
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (3.0, 10000.0), (2000.0, -10000.0)])
def test_entropy_matches_log_softmax_reference(shape, scale, offset):
    vocab = _module()
    torch.manual_seed(64)
    logits = (torch.randn(shape, device="cuda") * scale + offset).requires_grad_(True)
    with torch.no_grad():
        actual = vocab.entropy_from_logits(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        expected = -(log_probs.exp() * log_probs).sum(dim=-1)
    assert actual.shape == logits.shape[:-1]
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


def test_entropy_uniform_distribution_is_log_vocab():
    vocab = _module()
    width = 151936
    logits = torch.full((2, width), 10000.0, device="cuda")
    actual = vocab.entropy_from_logits(logits)
    expected = torch.full((2,), float(torch.tensor(width).log()), device="cuda")
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_entropy_handles_noncontiguous_leading_dimensions():
    vocab = _module()
    torch.manual_seed(65)
    logits = torch.randn((2, 3, 4103), device="cuda").transpose(0, 1)
    actual = vocab.entropy_from_logits(logits)
    log_probs = F.log_softmax(logits, dim=-1)
    expected = -(log_probs.exp() * log_probs).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


def test_cross_entropy_does_not_mutate_logits_or_targets():
    vocab = _module()
    torch.manual_seed(66)
    logits = torch.randn((3, 4103), device="cuda", requires_grad=True)
    targets = torch.tensor([1, -100, 4102], device="cuda")
    before_logits, before_targets = logits.detach().clone(), targets.clone()
    loss = vocab.cross_entropy(logits, targets)
    loss.backward()
    torch.testing.assert_close(logits, before_logits, rtol=0, atol=0)
    torch.testing.assert_close(targets, before_targets, rtol=0, atol=0)


def test_cross_entropy_empty_probability_tile_does_not_create_nan():
    vocab = _module()
    torch.manual_seed(67)
    logits = torch.randn((2, 4103), device="cuda")
    logits[:, :4096] = -torch.inf
    logits.requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([4096, 4102], device="cuda")
    actual = vocab.cross_entropy(logits, targets)
    expected = F.cross_entropy(ref_logits, targets)
    actual.backward()
    expected.backward()
    assert torch.isfinite(actual)
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-5, atol=3e-7)

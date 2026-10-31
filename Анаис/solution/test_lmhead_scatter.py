"""Scatter statistics preserve token mapping, prompt outputs and scaled gradients."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


@pytest.mark.parametrize("vocabulary", [3, 4103, 151936])
@pytest.mark.parametrize("write_ce", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_scatter_matches_dense_statistics_without_touching_other_rows(
        vocabulary, write_ce, strided):
    from solution.lmhead import BLOCK_V, lm_ce_kernel, lm_ce_scatter_kernel

    torch.manual_seed(327)
    count, tokens = 5, 13
    logits = torch.randn((count, vocabulary), device="cuda", dtype=torch.float16)
    # Exact ties exercise the first-index argmax rule, including a partial tile.
    logits[0].zero_()
    row_values = torch.tensor([9, 2, 12, 0, 5], device="cuda", dtype=torch.int64)
    rows = torch.full((2 * count,), -1, device="cuda", dtype=torch.int64)
    rows[::2] = row_values
    rows = rows[::2] if strided else row_values
    target_values = torch.randint(vocabulary, (tokens,), device="cuda")
    target_values[rows] = -100 if not write_ce else vocabulary - 1
    target_storage = torch.full((2 * tokens,), -101, device="cuda", dtype=torch.int64)
    target_storage[::2] = target_values
    targets = target_storage[::2] if strided else target_values
    selected_targets = targets.index_select(0, rows)
    reference_entropy = torch.empty(count, device="cuda")
    reference_argmax = torch.empty(count, device="cuda", dtype=torch.int64)
    reference_ce = torch.empty(count, device="cuda")
    reference_lse = torch.empty(count, device="cuda")
    lm_ce_kernel[(count,)](
        logits, selected_targets, reference_entropy, reference_argmax,
        reference_ce, reference_lse, logits.stride(0), vocabulary,
        BLOCK_V=BLOCK_V, num_warps=8)

    entropy = torch.full((tokens,), -7.0, device="cuda")
    argmax = torch.full((tokens,), -7, device="cuda", dtype=torch.int64)
    ce = torch.full((tokens,), -7.0, device="cuda")
    lse = torch.full((count,), -7.0, device="cuda")
    lm_ce_scatter_kernel[(count,)](
        logits, targets, rows, entropy, argmax, ce, lse, logits.stride(0),
        rows.stride(0), targets.stride(0), vocabulary, BLOCK_V=BLOCK_V,
        num_warps=8)

    selected = torch.zeros(tokens, device="cuda", dtype=torch.bool)
    selected[rows] = True
    torch.testing.assert_close(entropy[rows], reference_entropy, rtol=0, atol=0)
    assert torch.equal(argmax[rows], reference_argmax)
    assert argmax[rows[0]].item() == 0
    assert torch.all(entropy[~selected] == -7).item()
    assert torch.all(argmax[~selected] == -7).item()
    if write_ce:
        torch.testing.assert_close(ce[rows], reference_ce, rtol=0, atol=0)
        torch.testing.assert_close(lse, reference_lse, rtol=0, atol=0)
        assert torch.all(ce[~selected] == -7).item()
    else:
        assert torch.all(ce == -7).item()
        assert torch.all(lse == -7).item()


@pytest.mark.parametrize("vocabulary", [3, 4103])
def test_prompt_loss_pointers_can_alias_entropy_without_writing_it(vocabulary):
    from solution.lmhead import BLOCK_V, lm_ce_kernel, lm_ce_scatter_kernel

    torch.manual_seed(329)
    rows = torch.tensor([6, 0, 4], device="cuda", dtype=torch.int64)
    count, tokens = rows.numel(), 7
    logits = torch.randn((count, vocabulary), device="cuda", dtype=torch.float16)
    targets = torch.full((tokens,), -100, device="cuda", dtype=torch.int64)
    reference_entropy = torch.empty(count, device="cuda")
    reference_argmax = torch.empty(count, device="cuda", dtype=torch.int64)
    reference_ce, reference_lse = torch.empty_like(reference_entropy), torch.empty_like(reference_entropy)
    lm_ce_kernel[(count,)](
        logits, targets.index_select(0, rows), reference_entropy, reference_argmax,
        reference_ce, reference_lse, logits.stride(0), vocabulary,
        BLOCK_V=BLOCK_V, num_warps=8)
    entropy = torch.full((tokens,), -7.0, device="cuda")
    argmax = torch.full((tokens,), -7, device="cuda", dtype=torch.int64)
    # This is the production prompt call: no CE/LSE buffers exist. The masked
    # loss stores must not race with or overwrite any row's entropy output.
    lm_ce_scatter_kernel[(count,)](
        logits, targets, rows, entropy, argmax, entropy, entropy,
        logits.stride(0), rows.stride(0), targets.stride(0), vocabulary,
        BLOCK_V=BLOCK_V, num_warps=8)
    expected_entropy = torch.full_like(entropy, -7.0)
    expected_argmax = torch.full_like(argmax, -7)
    expected_entropy[rows], expected_argmax[rows] = reference_entropy, reference_argmax
    torch.testing.assert_close(entropy, expected_entropy, rtol=0, atol=0)
    assert torch.equal(argmax, expected_argmax)


@pytest.mark.parametrize("expected_scale", [None, 65536.0])
@pytest.mark.parametrize("actual_scale", [1.0, 65536.0])
def test_chunked_head_strided_targets_keep_native_metrics_and_gradient(
        monkeypatch, expected_scale, actual_scale):
    import solution.lmhead as lmhead
    from solution.test_lmhead_scaled import _assert_gradient

    monkeypatch.setattr(lmhead, "CHUNK", 3)
    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: expected_scale)
    torch.manual_seed(328)
    tokens, hidden, vocabulary = 11, 32, 4103
    x = torch.randn((tokens, hidden), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((vocabulary, hidden), device="cuda") * .05
    target_storage = torch.randint(vocabulary, (2 * tokens,), device="cuda")
    targets = target_storage[::2]
    targets[::3] = -100
    denom = (targets != -100).sum()
    logits = F.linear(reference.half(), weight.half()).float()
    expected = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum") / denom
    actual, stats = lmhead.lm_head_loss(x, weight, targets, denom)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    native_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(stats["entropy"], native_entropy, rtol=1e-4, atol=1e-5)
    assert torch.equal(stats["argmax"], logits.argmax(-1))
    assert torch.count_nonzero(stats["ce"][targets == -100]).item() == 0
    actual.backward(torch.tensor(actual_scale, device="cuda"))
    expected.backward(torch.tensor(actual_scale, device="cuda"))
    _assert_gradient(x.grad, reference.grad, targets == -100)

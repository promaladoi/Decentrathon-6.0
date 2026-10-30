"""Precision and alias-fallback checks for private half metrics."""

import inspect

import pytest
import torch
import torch.nn.functional as F


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def test_cpu_metadata_guards():
    from solution import half_metrics

    logits = torch.randn((3, 9, 17), dtype=torch.float16)
    labels = torch.ones((3, 9), dtype=torch.int64)
    assert not half_metrics.supports_half_entropy(logits)
    assert not half_metrics.supports_half_rows(logits, labels)


@CUDA
@pytest.mark.parametrize("shape", [(3, 9, 4103), (2, 17, 151936)])
def test_half_entropy_matches_promoted_logits(shape):
    from solution.half_metrics import entropy_half

    torch.manual_seed(211)
    logits = torch.randn(shape, device="cuda", dtype=torch.float16) * 3
    with torch.no_grad():
        log_probs = F.log_softmax(logits.float(), dim=-1)
        expected = -(log_probs.exp() * log_probs).sum(-1)
        actual = entropy_half(logits)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


@CUDA
@pytest.mark.parametrize("batch,sequence,vocabulary", [(3, 9, 4103), (4, 17, 151936)])
def test_batched_half_rewind_preserves_shift_counts_and_empty_rows(batch, sequence, vocabulary):
    from solution.half_metrics import row_losses_half

    torch.manual_seed(212)
    logits = torch.randn((batch, sequence, vocabulary), device="cuda", dtype=torch.float16) * 3
    labels = torch.randint(vocabulary, (batch, sequence), device="cuda")
    labels[0, :4] = -100
    labels[1, :] = -100
    labels[-1, -2:] = -100
    expected_values, expected_counts = [], []
    with torch.no_grad():
        for row in range(batch):
            targets = labels[row, 1:]
            losses = F.cross_entropy(logits[row, :-1].float(), targets,
                                     ignore_index=-100, reduction="none")
            mask = (targets != -100).to(torch.float32)
            count = mask.sum()
            expected_values.append(float((losses * mask).sum() / count.clamp(min=1)))
            expected_counts.append(int(count.item()))
        values, counts = row_losses_half(logits, labels)
    assert counts == expected_counts
    assert counts[1] == 0 and values[1] == 0.0
    torch.testing.assert_close(torch.tensor(values), torch.tensor(expected_values),
                               rtol=1e-6, atol=1e-6)


@CUDA
def test_aliases_only_use_half_path_in_active_scope(monkeypatch):
    from solution import half_metrics
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf

    expected_entropy = inspect.unwrap(sft_trainer.entropy_from_logits)
    expected_rows = inspect.unwrap(rewind_hf.row_losses)
    monkeypatch.setattr(sft_trainer, "entropy_from_logits", expected_entropy)
    monkeypatch.setattr(rewind_hf, "row_losses", expected_rows)
    active = [False]
    half_metrics.install(lambda: active[0])
    first_entropy, first_rows = sft_trainer.entropy_from_logits, rewind_hf.row_losses
    half_metrics.install(lambda: active[0])
    assert first_entropy is sft_trainer.entropy_from_logits
    assert first_rows is rewind_hf.row_losses
    logits = torch.randn((3, 9, 4103), device="cuda", dtype=torch.float16)
    labels = torch.randint(4103, (3, 9), device="cuda")
    labels[:, :3] = -100
    with torch.no_grad():
        torch.testing.assert_close(sft_trainer.entropy_from_logits(logits),
                                   expected_entropy(logits), rtol=0, atol=0)
        assert rewind_hf.row_losses(logits, labels) == expected_rows(logits, labels)
        active[0] = True
        torch.testing.assert_close(sft_trainer.entropy_from_logits(logits),
                                   half_metrics.entropy_half(logits), rtol=0, atol=0)
        assert rewind_hf.row_losses(logits, labels) == half_metrics.row_losses_half(logits, labels)


@CUDA
def test_active_scope_keeps_entropy_gradient_requests_on_original(monkeypatch):
    from solution import half_metrics
    from trl.trainer import sft_trainer

    original = inspect.unwrap(sft_trainer.entropy_from_logits)
    monkeypatch.setattr(sft_trainer, "entropy_from_logits", original)
    half_metrics.install(lambda: True)
    logits = torch.randn((2, 3, 17), device="cuda", dtype=torch.float16, requires_grad=True)
    actual, expected = sft_trainer.entropy_from_logits(logits), original(logits)
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

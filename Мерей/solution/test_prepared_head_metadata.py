"""Early head metadata keeps target shift, sorted indices and gradients exact."""

import pytest
import torch


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def _inputs(lengths, sequence, strided=False, ignored=False):
    batch = len(lengths)
    mask = (torch.arange(sequence, device="cuda")[None] <
            torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = torch.arange(batch * sequence, device="cuda", dtype=torch.int64).reshape(batch, sequence) % 97
    labels[:, ::3] = -100
    labels[mask == 0] = -100
    if ignored:
        labels.fill_(-100)
    if strided:
        wide_mask = torch.zeros((batch, 2 * sequence), device="cuda", dtype=mask.dtype)
        wide_labels = torch.full((batch, 2 * sequence), -999, device="cuda", dtype=torch.int64)
        wide_mask[:, ::2], wide_labels[:, ::2] = mask, labels
        mask, labels = wide_mask[:, ::2], wide_labels[:, ::2]
    return mask, labels


def _reference(mask, labels):
    batch, sequence = labels.shape
    rows = mask.reshape(-1).nonzero().squeeze(1)
    shifted = torch.cat((labels[:, 1:], labels.new_full((batch, 1), -100)), dim=1)
    targets = shifted.reshape(-1).index_select(0, rows)
    valid = targets != -100
    return rows, targets, valid.nonzero().squeeze(1), (~valid).nonzero().squeeze(1)


@requires_cuda
@pytest.mark.parametrize("lengths,sequence", [((1,), 1), ((5, 2), 5),
                                            ((29, 17, 8, 1), 29), ((512, 384, 7, 1), 512)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("ignored", [False, True])
def test_prepared_targets_and_both_index_lists_match_the_original(lengths, sequence, strided, ignored):
    from solution.batch_metadata import compact_batch_metadata

    mask, labels = _inputs(lengths, sequence, strided, ignored)
    original_mask, original_labels = mask.clone(), labels.clone()
    expected_rows, expected_targets, expected_labelled, expected_prompt = _reference(mask, labels)
    state = torch.cuda.get_rng_state()
    metadata = compact_batch_metadata(mask, labels)
    head = metadata["head"]
    assert head.is_current_source(labels)
    assert torch.equal(metadata["rows"], expected_rows)
    assert torch.equal(head.targets, expected_targets)
    assert torch.equal(head.labelled, expected_labelled)
    assert torch.equal(head.prompt, expected_prompt)
    assert head.labelled.untyped_storage().data_ptr() == head.prompt.untyped_storage().data_ptr()
    assert head.targets.untyped_storage().data_ptr() != head.order.untyped_storage().data_ptr()
    assert head.targets.untyped_storage().nbytes() + head.order.untyped_storage().nbytes() <= 32768
    assert torch.equal(torch.cuda.get_rng_state(), state)
    assert torch.equal(mask, original_mask) and torch.equal(labels, original_labels)


@requires_cuda
def test_shift_does_not_cross_sequences_and_keeps_invalid_targets_for_the_head_check():
    from solution.batch_metadata import compact_batch_metadata
    from solution.lmhead import _targets_in_bounds

    mask = torch.ones((2, 5), device="cuda", dtype=torch.int64)
    labels = torch.tensor([[999999, -100, 2, 3, 4], [888888, 5, 6, 7, 8]], device="cuda")
    head = compact_batch_metadata(mask, labels)["head"]
    assert head.targets[4].item() == head.targets[9].item() == -100
    assert _targets_in_bounds(head.targets, 97).item()
    labels[0, 2] = -101
    head = compact_batch_metadata(mask, labels)["head"]
    assert head.targets[1].item() == -101
    assert not _targets_in_bounds(head.targets, 97).item()


@requires_cuda
@pytest.mark.parametrize("changed", ["source", "targets", "order"])
def test_versioned_mutation_rejects_stale_preparation(changed):
    from solution.batch_metadata import compact_batch_metadata

    mask, labels = _inputs((7, 5), 7)
    head = compact_batch_metadata(mask, labels)["head"]
    assert head.is_current_source(labels)
    tensor = {"source": labels, "targets": head.targets, "order": head.order}[changed]
    tensor.reshape(-1)[0].add_(1)
    assert not head.is_current_source(labels)
    if changed != "source":
        assert head.indices_for(head.targets) is None


@requires_cuda
@pytest.mark.parametrize("expected_scale", [None, 65536.0])
@pytest.mark.parametrize("upstream", [1.0, 65536.0])
def test_prepared_head_avoids_nonzero_and_matches_current_loss_stats_and_gradient(
        monkeypatch, expected_scale, upstream):
    from solution.batch_metadata import compact_batch_metadata
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "CHUNK", 3)
    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: expected_scale)
    mask, labels = _inputs((9, 6), 9)
    head = compact_batch_metadata(mask, labels)["head"]
    torch.manual_seed(829)
    x = torch.randn((head.targets.numel(), 32), device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn((97, 32), device="cuda") * .05
    expected, ref_stats = lmhead.lm_head_loss(reference, weight, head.targets, None)

    def unexpected_nonzero(*args, **kwargs):
        raise AssertionError("prepared head must not synchronize through nonzero")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "nonzero", unexpected_nonzero)
        actual, stats = lmhead.lm_head_loss(x, weight, head.targets, None, prepared=head)
    assert "_prepared_head_metadata" not in stats
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ("entropy", "argmax", "ce", "valid"):
        torch.testing.assert_close(stats[key], ref_stats[key], rtol=0, atol=0)
    scale = torch.tensor(upstream, device="cuda")
    expected.backward(scale)
    actual.backward(scale)
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@requires_cuda
def test_retained_graph_and_input_target_mutation_do_not_corrupt_saved_indices(monkeypatch):
    from solution.batch_metadata import compact_batch_metadata
    import solution.lmhead as lmhead

    monkeypatch.setattr(lmhead, "_private_expected_loss_scale", lambda: 65536.0)
    mask, labels = _inputs((7, 5), 7)
    head = compact_batch_metadata(mask, labels)["head"]
    x = torch.randn((head.targets.numel(), 16), device="cuda", requires_grad=True)
    weight = torch.randn((97, 16), device="cuda") * .05
    loss, _ = lmhead.lm_head_loss(x, weight, head.targets, None, prepared=head)
    indices = loss.grad_fn.saved_tensors[1]
    initial_indices = indices.clone()
    head.targets.fill_(-100)
    # The old head saves its own target values; prepared targets must not share
    # the version counter of the separately saved sorted index storage.
    loss.backward(torch.tensor(65536.0, device="cuda"), retain_graph=True)
    first = x.grad.clone()
    x.grad = None
    loss.backward(torch.tensor(65536.0, device="cuda"))
    assert torch.equal(indices, initial_indices)
    torch.testing.assert_close(x.grad, first, rtol=0, atol=0)


def test_bulk_rewind_values_and_public_fallback_keep_the_original_python_types():
    from solution.lmhead import LazyLogits, make_row_losses

    labels = torch.tensor([[-100, 1, 1, -100], [-100, 1, -100, -100]])
    rows = torch.tensor([0, 1, 2, 4, 5])
    stats = {"ce": torch.tensor([2., 4., 0., 6., 8.])}
    logits = LazyLogits(2, 4, 97, rows, stats)
    sentinel = object()
    reader = make_row_losses(lambda logits, labels: sentinel)
    means, counts = reader(logits, labels)
    assert means == [3.0, 6.0] and counts == [2, 1]
    assert all(type(value) is float for value in means)
    assert all(type(value) is int for value in counts)
    assert reader(torch.zeros((2, 4, 97)), labels) is sentinel
    means, counts = reader(logits, labels.fill_(-100))
    assert means == [0.0, 0.0] and counts == [0, 0]

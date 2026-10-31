"""Stable causal targets/partitions must exactly preserve native indexing and loss."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


@pytest.mark.parametrize('lengths', [(1,), (17, 3), (31, 29, 1), (512, 384, 7, 1)])
@pytest.mark.parametrize('pattern', ['mixed', 'ignored', 'labelled'])
@pytest.mark.parametrize('strided', [False, True])
def test_targets_and_stable_partitions_match_native_exactly(lengths, pattern, strided):
    from solution.batch_metadata import compact_batch_metadata
    batch, sequence = len(lengths), max(lengths)
    mask = (torch.arange(sequence, device='cuda')[None] <
            torch.tensor(lengths, device='cuda')[:, None]).long()
    labels = torch.arange(batch * sequence, device='cuda').reshape(batch, sequence) % 4103
    labels = torch.where(mask.bool(), labels, -100)
    if pattern == 'mixed':
        labels[:, ::3] = -100
    elif pattern == 'ignored':
        labels.fill_(-100)
    if strided:
        wide_mask = torch.full((batch, sequence * 2), 99, device='cuda', dtype=torch.int64)
        wide_labels = torch.full_like(wide_mask, 99)
        wide_mask[:, ::2], wide_labels[:, ::2] = mask, labels
        mask, labels = wide_mask[:, ::2], wide_labels[:, ::2]
    originals = mask.clone(), labels.clone()
    rng = torch.cuda.get_rng_state()
    rows = mask.reshape(-1).nonzero().squeeze(1)
    shifted = torch.cat((labels[:, 1:], labels.new_full((batch, 1), -100)), dim=1)
    targets = shifted.reshape(-1).index_select(0, rows)
    valid = targets != -100
    result = compact_batch_metadata(mask, labels)
    assert result is not None
    assert torch.equal(result['targets'], targets)
    assert torch.equal(result['labelled'], valid.nonzero().squeeze(1))
    assert torch.equal(result['prompt'], (~valid).nonzero().squeeze(1))
    assert torch.equal(mask, originals[0]) and torch.equal(labels, originals[1])
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@pytest.mark.parametrize('expected_scale', [None, 1.0, 1024.0])
@pytest.mark.parametrize('pattern', ['mixed', 'ignored'])
def test_fused_partition_loss_statistics_and_hidden_gradients_are_identical(monkeypatch, expected_scale, pattern):
    import solution.lmhead as head
    from solution.batch_metadata import compact_batch_metadata
    torch.manual_seed(511)
    labels = torch.randint(4103, (2, 17), device='cuda')
    mask = torch.ones_like(labels)
    mask[1, 8:] = 0
    labels[1, 8:] = -100
    if pattern == 'ignored':
        labels.fill_(-100)
    else:
        labels[:, ::3] = -100
    metadata = compact_batch_metadata(mask, labels)
    targets = metadata['targets']
    x = (torch.randn((targets.numel(), 32), device='cuda') * .1).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    weight = torch.randn((4103, 32), device='cuda') * .03
    denominator = max(1, int((targets != -100).sum()))
    monkeypatch.setattr(head, '_private_expected_loss_scale', lambda: expected_scale)
    expected, reference_stats = head.lm_head_loss(ref_x, weight, targets, denominator)
    actual, stats = head.lm_head_loss(x, weight, targets, denominator,
                                     _partition_indices=(metadata['labelled'], metadata['prompt']))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats.keys() == reference_stats.keys()
    for name in stats:
        torch.testing.assert_close(stats[name], reference_stats[name], rtol=0, atol=0)
    scale = 1.0 if expected_scale is None else expected_scale
    upstream = torch.tensor(scale, device='cuda')
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)

# Team contribution by Merei; source 29419f43d9b9173d1ba5d6824a31589ad3b84df7.
"""Fused metadata keeps dense guards, row order, boundaries and RNG unchanged."""

import pytest
import torch
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def _batch(lengths, sequence=None, dtype=torch.int64, strided=False):
    batch = len(lengths)
    sequence = max(lengths) if sequence is None else sequence
    mask = (torch.arange(sequence, device="cuda")[None] <
            torch.tensor(lengths, device="cuda")[:, None]).to(dtype)
    labels = torch.arange(batch * sequence, device="cuda", dtype=torch.int64).reshape(batch, sequence)
    labels = torch.where(mask != 0, labels, -100)
    ids = torch.ones_like(labels)
    if strided:
        wide_mask = torch.full((batch, 2 * sequence), 9, device="cuda", dtype=dtype)
        wide_labels = torch.full((batch, 2 * sequence), 999, device="cuda", dtype=torch.int64)
        wide_mask[:, ::2], wide_labels[:, ::2] = mask, labels
        mask, labels = wide_mask[:, ::2], wide_labels[:, ::2]
    return ids, mask, labels


def _reference(mask, labels):
    valid = mask != 0
    counts = valid.sum(-1, dtype=torch.int32)
    sequence = mask.shape[1]
    prefix = torch.arange(sequence, device=mask.device)[None] < counts[:, None]
    accepted = (((mask == 0) | (mask == 1)).all() & (valid == prefix).all()
                & (counts > 0).all() & (valid | (labels == -100)).all()
                & (counts.sum() >= sequence))
    if not bool(accepted):
        return None
    return (torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32))),
            mask.reshape(-1).nonzero().squeeze(1))


@pytest.mark.parametrize("lengths,sequence", [((1,), 1), ((5, 3), 5), ((29, 17, 5), 29),
                                            ((512, 384, 7, 1), 512), ((16, 16, 16, 16), 16)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_metadata_matches_dense_compaction_exactly(lengths, sequence, dtype, strided):
    from solution.team_merei.batch_metadata import compact_batch_metadata
    _batch_metadata=lambda ids,mask,labels: compact_batch_metadata(mask,labels)

    ids, mask, labels = _batch(lengths, sequence, dtype, strided)
    before = (ids.clone(), mask.clone(), labels.clone())
    state = torch.cuda.get_rng_state()
    expected_cu, expected_rows = _reference(mask, labels)
    metadata = _batch_metadata(ids, mask, labels)
    assert metadata is not None and metadata["maximum"] == sequence
    assert metadata["rows"].dtype == torch.int64 and metadata["rows"].is_contiguous()
    assert metadata["cumulative"].dtype == torch.int32
    assert torch.equal(metadata["rows"], expected_rows)
    assert torch.equal(metadata["cumulative"], expected_cu)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    for actual, original in zip((ids, mask, labels), before):
        assert torch.equal(actual, original)


@pytest.mark.parametrize("kind", ["nonbinary", "negative", "left", "hole", "empty", "removed_label", "too_few"])
def test_invalid_metadata_has_identical_fallback_and_does_not_consume_rng(kind):
    from solution.team_merei.batch_metadata import compact_batch_metadata
    _batch_metadata=lambda ids,mask,labels: compact_batch_metadata(mask,labels)

    ids, mask, labels = _batch((5, 3), 5)
    if kind == "nonbinary":
        mask[1, 1] = 2
    elif kind == "negative":
        mask[1, 1] = -1
    elif kind == "left":
        mask[1] = torch.tensor([0, 0, 1, 1, 1], device="cuda")
    elif kind == "hole":
        mask[1] = torch.tensor([1, 0, 1, 1, 0], device="cuda")
    elif kind == "empty":
        mask[1].zero_()
    elif kind == "removed_label":
        labels[1, -1] = 42
    else:
        ids, mask, labels = _batch((1, 1), 5)
    state = torch.cuda.get_rng_state()
    assert _reference(mask, labels) is None
    assert _batch_metadata(ids, mask, labels) is None
    assert torch.equal(torch.cuda.get_rng_state(), state)


@pytest.mark.parametrize("lengths,sequence", [((7, 6, 5, 4, 3), 7), ((513, 257), 513)])
def test_larger_supported_dense_batches_keep_previous_metadata_fallback(lengths, sequence):
    from solution.team_merei.batch_metadata import compact_batch_metadata
    from solution.team_merei.batch_metadata import compact_batch_metadata
    _batch_metadata=lambda ids,mask,labels: compact_batch_metadata(mask,labels)

    ids, mask, labels = _batch(lengths, sequence)
    assert compact_batch_metadata(mask, labels) is NotImplemented
    expected_cu, _ = _reference(mask, labels)
    metadata = _batch_metadata(ids, mask, labels)
    assert metadata is NotImplemented


def test_compile_only_warmup_keeps_rng_state():
    from solution.team_merei.batch_metadata import warmup

    state = torch.cuda.get_rng_state()
    warmup()
    assert torch.equal(torch.cuda.get_rng_state(), state)

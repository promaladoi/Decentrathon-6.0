"""Validate one padded batch and compact its row indices in a single launch.

This only replaces metadata construction. It preserves row order, the original
right-prefix/label guards and the native attention sequence boundaries. One
small host read obtains validity and token counts, including shifted head-label
indices prepared before the layer loop instead of synchronizing after it. No RNG is used.
Larger or unsupported layouts return NotImplemented for the existing fallback.
"""

import torch
import triton
import triton.language as tl


MAX_BATCH = 4
MAX_SEQUENCE = 512


@triton.jit(do_not_specialize=["batch", "sequence", "mask_stride0", "mask_stride1",
                               "label_stride0", "label_stride1"])
def compact_batch_metadata_kernel(MASK, LABELS, ROWS, TARGETS, ORDER, CUMULATIVE, SUMMARY,
                                  batch, sequence, mask_stride0, mask_stride1,
                                  label_stride0, label_stride1,
                                  BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr):
    row = tl.arange(0, BLOCK_B)
    col = tl.arange(0, BLOCK_S)
    active = (row[:, None] < batch) & (col[None, :] < sequence)
    mask = tl.load(MASK + row[:, None] * mask_stride0 + col[None, :] * mask_stride1,
                   active, other=0)
    labels = tl.load(LABELS + row[:, None] * label_stride0 + col[None, :] * label_stride1,
                     active, other=-100)
    kept = active & (mask != 0)
    counts = tl.sum(kept.to(tl.int32), axis=1)
    binary = tl.min(tl.where(active, (mask == 0) | (mask == 1), True).to(tl.int32), axis=1)
    prefix = tl.min(tl.where(active, kept == (col[None, :] < counts[:, None]),
                             True).to(tl.int32), axis=1)
    labels_valid = tl.min(tl.where(active, kept | (labels == -100), True).to(tl.int32), axis=1)
    good_row = (row >= batch) | ((binary != 0) & (prefix != 0) &
                                 (labels_valid != 0) & (counts > 0))
    total = tl.sum(counts, axis=0)
    good = (tl.min(good_row.to(tl.int32), axis=0) != 0) & (total >= sequence)
    ends = tl.cumsum(counts, axis=0)
    starts = ends - counts
    tl.store(CUMULATIVE, 0)
    tl.store(CUMULATIVE + row + 1, ends, row < batch)
    # Prefix positions give every compact destination one writer even for an
    # invalid mask. Invalid batches return None before these rows are consumed.
    destination = starts[:, None] + col[None, :]
    source = row[:, None] * sequence + col[None, :]
    positions = (row[:, None] < batch) & (col[None, :] < counts[:, None])
    tl.store(ROWS + destination, source.to(tl.int64),
              positions)
    # The next label remains within its original sequence. In particular, a
    # last position never reads the following batch row's first label.
    shifted = tl.load(LABELS + row[:, None] * label_stride0 +
                       (col[None, :] + 1) * label_stride1,
                       active & (col[None, :] + 1 < sequence), other=-100)
    tl.store(TARGETS + destination, shifted, positions)
    flat_positions = tl.reshape(positions, (BLOCK_B * BLOCK_S,))
    flat_destination = tl.reshape(destination, (BLOCK_B * BLOCK_S,))
    flat_shifted = tl.reshape(shifted, (BLOCK_B * BLOCK_S,))
    labelled = flat_positions & (flat_shifted != -100)
    prompt = flat_positions & (flat_shifted == -100)
    labelled_count = tl.sum(labelled.to(tl.int32), 0)
    labelled_offset = tl.cumsum(labelled.to(tl.int32), 0) - 1
    prompt_offset = tl.cumsum(prompt.to(tl.int32), 0) - 1
    order_destination = tl.where(labelled, labelled_offset, labelled_count + prompt_offset)
    tl.store(ORDER + order_destination, flat_destination.to(tl.int64), flat_positions)
    tl.store(SUMMARY, good.to(tl.int32))
    tl.store(SUMMARY + 1, total)
    tl.store(SUMMARY + 2, labelled_count)


def _stamp(tensor):
    try:
        return (tensor._version, tensor.data_ptr(), tensor.device, tensor.dtype,
                tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
    except (RuntimeError, AttributeError):
        return None


class PreparedHeadMetadata:
    """Per-batch head indices, with no reuse across batches or autograd graphs."""

    def __init__(self, labels, targets, order, labelled_count):
        self.source_labels, self.targets, self.order = labels, targets, order
        self.labelled = order[:labelled_count]
        self.prompt = order[labelled_count:]
        self._source_stamp = _stamp(labels)
        self._target_stamp, self._order_stamp = _stamp(targets), _stamp(order)

    def _buffers_current(self):
        return (self._target_stamp is not None and self._order_stamp is not None
                and _stamp(self.targets) == self._target_stamp
                and _stamp(self.order) == self._order_stamp)

    def is_current_source(self, labels):
        return (labels is self.source_labels and self._source_stamp is not None
                and _stamp(labels) == self._source_stamp and self._buffers_current())

    def indices_for(self, targets):
        if targets is self.targets and self._buffers_current():
            return self.labelled, self.prompt
        return None


@torch.no_grad()
def compact_batch_metadata(attention_mask, labels):
    """Return metadata, None for invalid batches, or NotImplemented to fall back.

    The retained rows view owns at most 4 * 512 * 8 bytes. This small capacity
    avoids a second allocation/copy after the single explicit synchronization.
    Targets and their combined label/prompt index storage each have the same
    capacity; they are per batch and add at most 32 KiB before the layer loop.
    Overall GPU peak is still qualified with the unchanged contest harness.
    """
    if (not isinstance(attention_mask, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or not attention_mask.is_cuda or not labels.is_cuda
            or attention_mask.device != labels.device
            or attention_mask.dtype not in (torch.bool, torch.int32, torch.int64)
            or labels.dtype != torch.int64 or attention_mask.ndim != 2
            or labels.shape != attention_mask.shape
            or not 0 < attention_mask.shape[0] <= MAX_BATCH
            or not 0 < attention_mask.shape[1] <= MAX_SEQUENCE):
        return NotImplemented
    batch, sequence = attention_mask.shape
    rows_storage = torch.empty(batch * sequence, device=attention_mask.device, dtype=torch.int64)
    targets_storage = torch.empty_like(rows_storage)
    # One allocation contains the two sorted index lists. Keep targets in a
    # separate storage so their version counter cannot invalidate saved indices.
    order_storage = torch.empty_like(rows_storage)
    cumulative = torch.empty(batch + 1, device=attention_mask.device, dtype=torch.int32)
    summary = torch.empty(3, device=attention_mask.device, dtype=torch.int32)
    compact_batch_metadata_kernel[(1,)](
        attention_mask, labels, rows_storage, targets_storage, order_storage,
        cumulative, summary, batch, sequence,
        *attention_mask.stride(), *labels.stride(),
        BLOCK_B=MAX_BATCH, BLOCK_S=MAX_SEQUENCE, num_warps=4)
    valid, count, labelled_count = summary.tolist()
    if not valid:
        return None
    head = PreparedHeadMetadata(labels, targets_storage[:count], order_storage[:count], labelled_count)
    return {"cumulative": cumulative, "maximum": sequence, "rows": rows_storage[:count],
            "head": head}


def warmup():
    """Compile the task's int64-mask signature without executing metadata work."""
    from .warmup import load_compiled

    load_compiled(compact_batch_metadata_kernel.warmup(
        torch.int64, torch.int64, torch.int64, torch.int64, torch.int64, torch.int32, torch.int32,
        MAX_BATCH, MAX_SEQUENCE, MAX_SEQUENCE, 1, MAX_SEQUENCE, 1,
        BLOCK_B=MAX_BATCH, BLOCK_S=MAX_SEQUENCE, grid=(1,), num_warps=4))

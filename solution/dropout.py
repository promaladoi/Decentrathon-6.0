"""Real-row dropout with native full-shape CUDA RNG and packed saved masks.

The existing large-shape dispatch in unpad.py is unchanged. RNG remains PyTorch's native
dropout on a fresh, contiguous FP32 [batch, sequence, hidden] allocation. Its
value output is discarded; the used mask is gathered while applying it to the
actual unpadded values. The backward reads only the saved real-row mask bits,
eight per byte. Packing shares the gather/apply launch; it does not draw RNG.
No RNG state is read, reset, or manually advanced by this production path.

The original Triton kernels below implement gather plus two multiplications;
they do not implement/copy Philox or a third-party dropout kernel. Contracts:
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu
https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml
"""

import struct

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

BLOCK = 1024


@triton.jit
def real_dropout_gather_kernel(X, FullMask, Rows, Out, PackedMask, N, FULL_ROWS,
                               WIDTH: tl.constexpr, SCALE: tl.constexpr,
                               BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    row = tl.load(Rows + offset // WIDTH, valid, other=0)
    valid_row = valid & (row >= 0) & (row < FULL_ROWS)
    kept = tl.load(FullMask + row * WIDTH + offset % WIDTH, valid_row, other=False)
    value = tl.load(X + offset, valid, other=0.0).to(tl.float32)
    # Native forward multiplies in this order. A where(mask, value, 0) would
    # change dropped NaN/Inf and signed-zero behavior.
    masked = value * kept.to(tl.float32)
    tl.store(Out + offset, masked * SCALE, valid)
    # Consecutive flat elements share a byte, even when WIDTH is odd and a
    # byte straddles two rows. BLOCK is a multiple of eight: each byte has
    # exactly one writer. Masked tail elements contribute zero bits.
    bits = tl.reshape(kept.to(tl.int32), (BLOCK // 8, 8))
    packed = tl.sum(bits * (1 << tl.arange(0, 8)[None, :]), axis=1)
    byte = tl.program_id(0) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    tl.store(PackedMask + byte, packed.to(tl.uint8), byte < (N + 7) // 8)


@triton.jit
def real_dropout_backward_kernel(Grad, PackedMask, DX, N, SCALE: tl.constexpr,
                                 BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    byte = tl.load(PackedMask + offset // 8, valid, other=0).to(tl.int32)
    kept = ((byte >> (offset % 8)) & 1).to(tl.float32)
    grad = tl.load(Grad + offset, valid, other=0.0).to(tl.float32)
    # The original index_select backward first index_adds into positive zeros.
    # With unique rows this changes only an incoming negative zero's sign.
    grad = tl.where(grad == 0.0, 0.0, grad)
    # Native masked_scale evaluates mask * grad * scale, without a select.
    masked = kept * grad
    tl.store(DX + offset, masked * SCALE, valid)


def _float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def _scales(p):
    # CUDA native forward receives keep_probability converted to accscalar_t
    # before reciprocal; autograd passes backward's reciprocal as a double.
    keep = 1.0 - float(p)
    return _float32(1.0 / _float32(keep)), _float32(1.0 / keep)


def _supported(x, rows, batch, seq_len, p, training, rows_from_nonzero):
    return (rows_from_nonzero is True and training is True
            and type(batch) is int and type(seq_len) is int and batch > 0 and seq_len > 0
            and isinstance(p, (float, int)) and 0 < p < 1
            and isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32
            and x.ndim == 2 and x.numel() > 0 and x.shape[1] > 0 and x.is_contiguous()
            and isinstance(rows, torch.Tensor) and rows.device == x.device
            and rows.dtype == torch.int64 and rows.ndim == 1 and rows.is_contiguous()
            and rows.numel() == x.shape[0] and rows.numel() <= batch * seq_len)


def reference_dropout(x, rows, batch, seq_len, p=0.05, training=True):
    """The dense-pad/native-dropout/gather reference, including its autograd."""
    padded = x.new_zeros((batch * seq_len, x.shape[-1])).index_copy(0, rows, x)
    padded = padded.view(batch, seq_len, x.shape[-1])
    return F.dropout(padded, p=p, training=training).view(
        batch * seq_len, x.shape[-1]).index_select(0, rows)


class _RealRowDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, batch, seq_len, p):
        forward_scale, backward_scale = _scales(p)
        dummy = torch.empty((batch, seq_len, x.shape[1]), device=x.device,
                            dtype=torch.float32, memory_format=torch.contiguous_format)
        # Both the original index_copy allocation and this allocation start
        # at an allocator-aligned address. This keeps native vectorization
        # and its per-element RNG mapping identical under the harness allocator.
        if dummy.data_ptr() % 16:
            raise RuntimeError('native dropout candidate requires aligned CUDA allocation')
        discarded, full_mask = torch.ops.aten.native_dropout.default(dummy, float(p), True)
        del discarded, dummy
        output = torch.empty_like(x)
        n = x.numel()
        packed_mask = torch.empty((triton.cdiv(n, 8),), dtype=torch.uint8, device=x.device)
        real_dropout_gather_kernel[(triton.cdiv(n, BLOCK),)](
            x, full_mask, rows, output, packed_mask, n, batch * seq_len,
            x.shape[1], forward_scale, BLOCK, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(packed_mask)
        ctx.backward_scale = backward_scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        packed_mask, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dx = torch.empty_like(grad_output)
        n = grad_output.numel()
        real_dropout_backward_kernel[(triton.cdiv(n, BLOCK),)](
            grad_output, packed_mask, dx, n, ctx.backward_scale, BLOCK,
            num_warps=4, enable_fp_fusion=False)
        return dx, None, None, None, None


def real_row_dropout(x, rows, batch, seq_len, p=0.05, training=True, *, rows_from_nonzero=False):
    """Dropout equivalent to the dense padded operation for trusted real rows.

    Fast-path caller contract: rows_from_nonzero=True means rows came from
    the matching attention_mask.reshape(-1).nonzero().squeeze(1), hence they
    are unique and in range. That invariant is not checked with a hot-loop
    device synchronization. Other callers use the reference operation.
    This function does not inspect or bypass nn.Module hooks.
    """
    if not _supported(x, rows, batch, seq_len, p, training, rows_from_nonzero):
        return reference_dropout(x, rows, batch, seq_len, p, training)
    return _RealRowDropout.apply(x, rows, batch, seq_len, p)


def warmup(width=1536, p=0.05):
    """Compile only the original gather/apply kernels; never draw random bits."""
    from .warmup import load_compiled

    forward, backward = _scales(p)
    load_compiled(real_dropout_gather_kernel.warmup(
        torch.float32, torch.bool, torch.int64, torch.float32, torch.uint8,
        16 * width, 16, width, forward, BLOCK, grid=(1,),
        num_warps=4, enable_fp_fusion=False))
    load_compiled(real_dropout_backward_kernel.warmup(
        torch.float32, torch.uint8, torch.float32, 16 * width, backward, BLOCK,
        grid=(1,), num_warps=4, enable_fp_fusion=False))

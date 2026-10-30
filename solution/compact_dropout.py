"""Keep PyTorch's full-shape CUDA dropout draw and compact its backward.

The native operator is the CUDA implementation functional.dropout dispatches to.
Retain its boolean mask directly instead of filling ones and inferring a second
mask from the discarded scaled output. Only real rows are multiplied and saved.
No custom RNG, changed probability, batch packing or skipped random draws.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def compact_dropout_rows(X, MASK, ROWS, OUT, N, H: tl.constexpr,
                         SCALE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // H
    col = i % H
    original_row = tl.load(ROWS + row, i < N, other=0)
    mask = tl.load(MASK + original_row * H + col, i < N, other=0).to(tl.float32)
    value = tl.load(X + i, i < N, other=0.0).to(tl.float32)
    tl.store(OUT + i, (value * mask) * SCALE, i < N)


class _CompactDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, batch, seq_len, p):
        # The kernel reads ROWS + row, so it needs unit-stride indices. A
        # slice such as arange(...)[::3] is a view with stride 3; reading it as
        # dense indices silently selects other tokens' masks. Normalize once
        # and retain that same index tensor for backward. Production nonzero()
        # rows already have unit stride, so this is normally a no-op.
        rows = rows.contiguous()
        mask = _draw_full_shape_mask(x, batch, seq_len, p)
        out = torch.empty_like(x)
        compact_dropout_rows[(triton.cdiv(x.numel(), 1024),)](
            x, mask, rows, out, x.numel(), x.shape[-1], 1.0 / (1.0 - p),
            1024, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(mask, rows)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, grad):
        mask, rows = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(grad)
        compact_dropout_rows[(triton.cdiv(grad.numel(), 1024),)](
            grad, mask, rows, dx, grad.numel(), grad.shape[-1], 1.0 / (1.0 - ctx.p),
            1024, num_warps=4, enable_fp_fusion=False)
        return dx, None, None, None, None


def compact_dropout(x, rows, batch, seq_len, p):
    return _CompactDropout.apply(x, rows, batch, seq_len, p)


def _draw_full_shape_mask(x, batch, seq_len, p):
    # CUDA native_dropout's mask/RNG depend on shape, dtype and layout, not
    # input values. Its numeric output is discarded. Use the same contiguous
    # full padded layout and normal PyTorch allocator as functional.dropout.
    dummy = torch.empty((batch, seq_len, x.shape[-1]), device=x.device, dtype=x.dtype)
    discarded, mask = torch.ops.aten.native_dropout.default(dummy, p, True)
    del discarded, dummy
    return mask


def warmup(hidden=1536, p=.05):
    """Compile both production storage types with a runtime token count."""
    for dtype in (torch.float16, torch.float32):
        compact_dropout_rows.warmup(
            dtype, torch.bool, torch.int64, dtype, 16 * hidden,
            hidden, 1.0 / (1.0 - p), 1024, grid=(1,), num_warps=4,
            enable_fp_fusion=False)


def patch_compact_dropout():
    from .team_vladimir.unpad import _Batch
    original = _Batch.dropout
    if getattr(original, '_compact_dropout', False):
        return

    def dropout(self, module, x):
        if (type(module) is torch.nn.Dropout and module.training and 0 < module.p < 1
                and x.is_cuda and x.is_contiguous() and x.dtype in (torch.float16, torch.float32)
                and x.ndim == 2 and x.numel() > 0 and not module.inplace
                and not module._forward_hooks and not module._forward_pre_hooks
                and not module._backward_hooks):
            return compact_dropout(x, self.rows, self.batch, self.seq_len, module.p)
        return original(self, module, x)

    dropout._compact_dropout = True
    _Batch.dropout = dropout

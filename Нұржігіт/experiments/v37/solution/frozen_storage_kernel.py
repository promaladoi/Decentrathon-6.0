"""Pack a frozen fp32 tensor and prove its fp16 values lossless in one CUDA pass."""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit(do_not_specialize=['N'])
def frozen_half_copy_checked(X, Y, EXACT, N, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, other=0.0)
    compact = value.to(tl.float16)
    exact = (compact.to(tl.float32) == value) & (tl.abs(value) < float('inf'))
    valid = tl.min(tl.where(i < N, exact, True).to(tl.int32), 0)
    tl.store(Y + i, compact, i < N)
    tl.store(EXACT + block, valid)


def exact_half_copy_cuda(weight):
    assert weight.is_cuda and weight.dtype == torch.float32 and weight.is_contiguous()
    count = weight.numel()
    packed = torch.empty_like(weight, dtype=torch.float16)
    flags = torch.empty(triton.cdiv(count, BLOCK), device=weight.device, dtype=torch.int32)
    frozen_half_copy_checked[(flags.numel(),)](weight, packed, flags, count,
                                               BLOCK, num_warps=4)
    return packed if flags.all().item() else None


def warmup():
    frozen_half_copy_checked.warmup(torch.float32, torch.float16, torch.int32,
                                    1024, BLOCK, grid=(1,), num_warps=4)

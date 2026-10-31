"""Sum frozen-projection input gradients after their unchanged FP16 GEMMs.

The native path casts each GEMM output to FP32 before adding it. This kernel
performs those same casts and ordered additions in one pass; it never changes
the GEMM, accumulation order or final caller-selected gradient dtype.
"""

import torch
import triton
import triton.language as tl


BLOCK = 1024


@triton.jit(do_not_specialize=["N"])
def sum_half_gradients_kernel(A, B, C, OUT, N, THREE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    first = tl.load(A + index, valid, other=0.0).to(tl.float32)
    second = tl.load(B + index, valid, other=0.0).to(tl.float32)
    result = first + second
    if THREE:
        third = tl.load(C + index, valid, other=0.0).to(tl.float32)
        result = result + third
    tl.store(OUT + index, result, valid)


def _torch_sum(parts):
    result = parts[0].float()
    for part in parts[1:]:
        result = result + part.float()
    return result


def sum_half_gradients(parts):
    """Return ``(a.float() + b.float()) [+ c.float()]`` without mutating inputs.

    Higher-order differentiation keeps the ordinary torch graph. The fast
    path is restricted to the ordinary no-grad backward and matching dense
    CUDA half tensors; callers retain their original final dtype conversion.
    """
    if not isinstance(parts, (tuple, list)):
        raise TypeError("gradient parts must be a tuple or list")
    if len(parts) not in (2, 3):
        raise ValueError("expected two or three gradient parts")
    if not all(isinstance(part, torch.Tensor) for part in parts):
        raise TypeError("gradient parts must be tensors")
    first = parts[0]
    if (torch.is_grad_enabled() or any(part.requires_grad for part in parts)
            or not first.is_cuda or first.dtype != torch.float16 or first.numel() == 0
            or not first.is_contiguous()
            or any(part.device != first.device or part.dtype != torch.float16
                   or part.shape != first.shape or not part.is_contiguous()
                   for part in parts[1:])):
        return _torch_sum(parts)
    output = torch.empty_like(first, dtype=torch.float32)
    sum_half_gradients_kernel[(triton.cdiv(first.numel(), BLOCK),)](
        first, parts[1], parts[2] if len(parts) == 3 else first,
        output, first.numel(), THREE=len(parts) == 3, BLOCK=BLOCK,
        num_warps=4, enable_fp_fusion=False)
    return output


def warmup():
    """Compile both arities without launching arithmetic or consuming RNG."""
    from .warmup import load_compiled

    for three in (False, True):
        load_compiled(sum_half_gradients_kernel.warmup(
            torch.float16, torch.float16, torch.float16, torch.float32,
            16 * 1536, THREE=three, BLOCK=BLOCK, grid=(1,),
            num_warps=4, enable_fp_fusion=False))

"""Original fused NF4 unpack + optional double-quantized scale reconstruction.

The interface follows bitsandbytes 0.50.2 functional.dequantize_4bit and
autograd.MatMul4Bit.backward. The matrix product remains bitsandbytes/PyTorch.
This kernel writes the same dense [out_features, in_features] weight used by
the input-gradient matmul, without a separate decoded-absmax temporary.
"""

import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def soup_nf4_unpack_kernel(Packed, Absmax, Code4, Absmax2, Code8, Offset, Out,
                           N, QUANT_BLOCK: tl.constexpr,
                           NESTED: tl.constexpr, SCALE_BLOCK: tl.constexpr,
                           BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < N
    byte = tl.load(Packed + position // 2, valid, other=0).to(tl.int32)
    index = tl.where(position % 2 == 0, byte >> 4, byte & 15)
    value = tl.load(Code4 + index).to(tl.float32)
    quant_block = position // QUANT_BLOCK
    if NESTED:
        scale_index = tl.load(Absmax + quant_block, valid, other=0).to(tl.int32)
        scale_code = tl.load(Code8 + scale_index).to(tl.float32)
        scale_max = tl.load(Absmax2 + quant_block // SCALE_BLOCK,
                            valid, other=0).to(tl.float32)
        # The stock path decodes the 8-bit scales, then adds the offset in a
        # separate float32 operation. Disable FMA at launch to retain this.
        scale = scale_code * scale_max
        scale = scale + tl.load(Offset).to(tl.float32)
    else:
        scale = tl.load(Absmax + quant_block, valid, other=0).to(tl.float32)
    tl.store(Out + position, value * scale, valid)


def _tensor(value, device, dtype, minimum):
    return (isinstance(value, torch.Tensor) and value.device == device
            and value.dtype == dtype and value.is_contiguous()
            and value.numel() >= minimum and not value.requires_grad)


def _supported(packed, state, out=None):
    if (out is not None or not isinstance(packed, torch.Tensor)
            or not packed.is_cuda or packed.dtype != torch.uint8
            or not packed.is_contiguous() or packed.ndim != 2
            or (packed.shape[0] != 1 and packed.shape[1] != 1)
            or state is None or getattr(state, "quant_type", None) != "nf4"):
        return False
    shape = getattr(state, "shape", None)
    dtype = getattr(state, "dtype", None)
    block = getattr(state, "blocksize", None)
    if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
            or not all(isinstance(size, int) and size > 0 for size in shape)
            or dtype not in (torch.float16, torch.float32)
            or block not in (32, 64, 128, 256, 512, 1024, 2048, 4096)):
        return False
    elements = math.prod(shape)
    scales = triton.cdiv(elements, block)
    if packed.numel() != triton.cdiv(elements, 2):
        return False
    if not _tensor(getattr(state, "code", None), packed.device, torch.float32, 16):
        return False
    if not getattr(state, "nested", False):
        return _tensor(getattr(state, "absmax", None), packed.device, torch.float32, scales)
    second = getattr(state, "state2", None)
    return (second is not None and getattr(second, "blocksize", None) == 256
            and not getattr(second, "nested", False)
            and getattr(second, "dtype", None) == torch.float32
            and _tensor(getattr(state, "absmax", None), packed.device, torch.uint8, scales)
            and _tensor(getattr(second, "code", None), packed.device, torch.float32, 256)
            and _tensor(getattr(second, "absmax", None), packed.device,
                        torch.float32, triton.cdiv(scales, 256))
            and _tensor(getattr(state, "offset", None), packed.device, torch.float32, 1)
            and state.offset.numel() == 1)


def dequantize_nf4(packed, state):
    """Fast path for validated CUDA NF4 QuantState objects, including BC transpose."""
    if not _supported(packed, state):
        raise ValueError("unsupported input for the solution NF4 fast path")
    dense = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    nested = bool(state.nested)
    if nested:
        second_max, code8, offset = state.state2.absmax, state.state2.code, state.offset
    else:
        # These pointers are eliminated by the non-nested compile-time branch.
        second_max = code8 = offset = state.absmax
    elements = dense.numel()
    soup_nf4_unpack_kernel[(triton.cdiv(elements, 1024),)](
        packed, state.absmax, state.code, second_max, code8, offset, dense,
        elements, state.blocksize, nested, 256, 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    # This is bitsandbytes' documented compatibility orientation. Its own
    # MatMul4Bit normalizes B to a packed column before backward calls us.
    return dense.t() if packed.shape[0] == 1 else dense


def patch_nf4_dequant():
    """Patch the Python lookup really used by MatMul4Bit.backward; preserve fallbacks."""
    import bitsandbytes.functional as functional

    original = functional.dequantize_4bit
    if getattr(original, "_soup_nf4_patch", False):
        return

    @functools.wraps(original)
    def dequantize(packed, quant_state=None, absmax=None, out=None,
                   blocksize=None, quant_type="fp4"):
        # With QuantState, the stock function takes dtype/blocksize/type from
        # that state; its standalone defaults (including "fp4") are ignored.
        if not _supported(packed, quant_state, out):
            return original(packed, quant_state, absmax, out, blocksize, quant_type)
        return dequantize_nf4(packed, quant_state)

    dequantize._soup_nf4_patch = True
    functional.dequantize_4bit = dequantize

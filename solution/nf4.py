"""Original NF4 decoder, including the nested scale decode in a single pass.

The storage layout and QuantState semantics are from bitsandbytes 0.50.2
functional.dequantize_4bit. Codebook values are read from the actual quantization
state: no copied CUDA/Triton kernel or hard-coded model weights are included.
"""

import math

import torch
import triton
import triton.language as tl

_codebooks = {}


@triton.jit
def decode_nf4_weights(packed, scales, code4, scale_code, scale_absmax,
                       offset, out, N: tl.constexpr, GROUP: tl.constexpr,
                       SCALE_GROUP: tl.constexpr, NESTED: tl.constexpr,
                       BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + i // 2, i < N, other=0).to(tl.int32)
    code = tl.where(i % 2 == 0, byte >> 4, byte & 15)
    value = tl.load(code4 + code).to(tl.float32)
    group = i // GROUP
    if NESTED:
        quantized_scale = tl.load(scales + group, i < N, other=0).to(tl.int32)
        scale_value = tl.load(scale_code + quantized_scale).to(tl.float32)
        outer_scale = tl.load(scale_absmax + group // SCALE_GROUP, i < N, other=0).to(tl.float32)
        scale = scale_value * outer_scale + tl.load(offset).to(tl.float32)
    else:
        scale = tl.load(scales + group, i < N, other=0).to(tl.float32)
    tl.store(out + i, value * scale, i < N)


def dequantize_nf4(packed, state, out=None):
    if out is None:
        out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    nested = state.nested
    decode_nf4_weights[(triton.cdiv(math.prod(state.shape), 1024),)](
        packed, state.absmax, state.code,
        state.state2.code if nested else state.code,
        state.state2.absmax if nested else state.absmax,
        state.offset if nested else state.absmax,
        out, math.prod(state.shape), state.blocksize,
        state.state2.blocksize if nested else 1, nested, 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return out.t() if packed.shape[0] == 1 else out


def nf4_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                        bias=None, absmax_8bit=None, absmax_code=None,
                        absmax_offset=None):
    """Drop-in NF4 decode+linear for bnb's existing CUDA fallback call site."""
    import bitsandbytes.functional as functional

    key = (A.device.type, A.device.index)
    if key not in _codebooks:
        # Only the fixed NF4 format codebook is cached, never model weights.
        _codebooks[key] = functional.get_4bit_type("nf4", device=A.device)
    code = _codebooks[key]
    dense = torch.empty(shapeB, dtype=A.dtype, device=A.device)
    nested = absmax_8bit is not None
    decode_nf4_weights[(triton.cdiv(math.prod(shapeB), 1024),)](
        B, absmax_8bit if nested else absmax, code,
        absmax_code if nested else code, absmax,
        absmax_offset if nested else absmax,
        dense, math.prod(shapeB), blocksize, 256 if nested else 1, nested, 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return torch.nn.functional.linear(A, dense, bias)


def patch_nf4_decoder():
    import bitsandbytes.functional as functional

    if getattr(functional.dequantize_4bit, "_soup_team_nf4", False):
        return
    original = functional.dequantize_4bit

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None,
                       blocksize=None, quant_type="fp4"):
        state = quant_state
        supported = (
            isinstance(A, torch.Tensor) and A.is_cuda and A.is_contiguous()
            and A.dtype == torch.uint8 and A.ndim == 2 and state is not None
            and state.quant_type == "nf4" and state.code is not None
            and state.dtype in (torch.float16, torch.float32)
            and len(state.shape) == 2 and math.prod(state.shape) > 0
            and state.absmax.is_cuda and state.code.is_cuda
            and state.code.dtype == torch.float32 and state.code.numel() == 16
            and (not state.nested or (
                state.state2 is not None and state.offset.is_cuda
                and state.state2.code.is_cuda and state.state2.absmax.is_cuda
                and not state.state2.nested
            ))
            and (out is None or (
                out.is_contiguous() and tuple(out.shape) == tuple(state.shape)
                and out.dtype == state.dtype and out.device == A.device
            ))
        )
        if supported:
            return dequantize_nf4(A, state, out)
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._soup_team_nf4 = True
    functional.dequantize_4bit = dequantize_4bit

    from bitsandbytes.backends.cuda import ops

    if getattr(ops._dequant_linear_fallback, "_soup_team_nf4", False):
        return
    original_linear = ops._dequant_linear_fallback

    def linear(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
               absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if (A.is_cuda and A.dtype in (torch.float16, torch.float32)
                and B.dtype == torch.uint8 and B.is_contiguous()
                and quant_type == "nf4" and len(shapeB) == 2
                and math.prod(shapeB) > 0):
            return nf4_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                                       bias, absmax_8bit, absmax_code, absmax_offset)
        return original_linear(A, B, shapeB, absmax, blocksize, quant_type,
                               bias, absmax_8bit, absmax_code, absmax_offset)

    linear._soup_team_nf4 = True
    ops._dequant_linear_fallback = linear

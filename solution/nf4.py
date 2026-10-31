"""Exact NF4 decoding with the team's frozen-matmul integration.

Integration: teammate Vladimir, source commit 987da36e1efa90652e1a7f8c48afef50685dbc40.
Decode arithmetic/validation: Merey's verified original NF4 kernel, commit
b7c406dea7dbdc379149910efe044dccf155d083. The two original contributions are
combined here; neither is presented as third-party code written by somebody else.

Pinned bitsandbytes 0.50.2 selects custom GEMM for some small-M shapes and
dequantize+F.linear for the rest. Preserve that choice, replacing only the
fallback's decode. Backward first rounds to QuantState.dtype, as its Python
dequantize_4bit does, before converting to grad_output.dtype.
"""

import functools
import math

import torch
import triton
import triton.language as tl

BLOCK = 1024
_resident_cache = None


@triton.jit
def nf4_dequant_kernel(Packed, Absmax, Code4, Absmax2, Code8, Offset, Out,
                       N, QUANT_BLOCK: tl.constexpr, NESTED: tl.constexpr,
                       SCALE_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
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
        scale = scale_code * scale_max
        scale = scale + tl.load(Offset).to(tl.float32)
    else:
        scale = tl.load(Absmax + quant_block, valid, other=0).to(tl.float32)
    tl.store(Out + position, value * scale, valid)


def _tensor(value, device, dtype, minimum):
    return (isinstance(value, torch.Tensor) and value.device == device
            and value.dtype == dtype and value.is_contiguous()
            and value.numel() >= minimum and not value.requires_grad)


def supported(state):
    shape = getattr(state, 'shape', None)
    dtype = getattr(state, 'dtype', None)
    block = getattr(state, 'blocksize', None)
    code = getattr(state, 'code', None)
    if (getattr(state, 'quant_type', None) != 'nf4'
            or not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
            or not all(isinstance(size, int) and size > 0 for size in shape)
            or dtype not in (torch.float16, torch.float32)
            or block not in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            or not isinstance(code, torch.Tensor) or not code.is_cuda):
        return False
    device = code.device
    scales = triton.cdiv(math.prod(shape), block)
    if not _tensor(code, device, torch.float32, 16) or code.numel() != 16:
        return False
    if not getattr(state, 'nested', False):
        return _tensor(getattr(state, 'absmax', None), device, torch.float32, scales)
    second = getattr(state, 'state2', None)
    return (second is not None and getattr(second, 'blocksize', None) == 256
            and not getattr(second, 'nested', False)
            and getattr(second, 'dtype', None) == torch.float32
            and _tensor(getattr(state, 'absmax', None), device, torch.uint8, scales)
            and _tensor(getattr(second, 'code', None), device, torch.float32, 256)
            and _tensor(getattr(second, 'absmax', None), device,
                        torch.float32, triton.cdiv(scales, 256))
            and _tensor(getattr(state, 'offset', None), device, torch.float32, 1)
            and state.offset.numel() == 1)


def _packed_supported(packed, state):
    return (supported(state) and isinstance(packed, torch.Tensor)
            and packed.is_cuda and packed.device == state.code.device
            and packed.dtype == torch.uint8 and packed.is_contiguous()
            and packed.numel() == triton.cdiv(math.prod(state.shape), 2)
            and not packed.requires_grad)


def _decode(packed, state, dtype):
    """Raw decode to the destination dtype, as CUDA forward fallback does."""
    if not _packed_supported(packed, state) or dtype not in (torch.float16, torch.float32):
        raise ValueError('unsupported CUDA NF4 input')
    dense = torch.empty(state.shape, dtype=dtype, device=packed.device)
    nested = bool(state.nested)
    if nested:
        second_max, code8, offset = state.state2.absmax, state.state2.code, state.offset
    else:
        second_max = code8 = offset = state.absmax
    elements = dense.numel()
    nf4_dequant_kernel[(triton.cdiv(elements, BLOCK),)](
        packed, state.absmax, state.code, second_max, code8, offset, dense,
        elements, state.blocksize, nested, 256, BLOCK,
        num_warps=4, enable_fp_fusion=False)
    return dense


def dequantize_nf4(packed, state, dtype):
    """Canonical [N,K], with native functional-decode then cast semantics.

    Unlike functional.dequantize_4bit's compatibility transpose convention,
    this team's existing API always returns the canonical weight orientation.
    """
    return _decode(packed, state, state.dtype).to(dtype)


def enable_resident_weights(max_bytes=2_400_000_000):
    """Enable bounded, per-process storage for internal frozen consumers."""
    global _resident_cache
    from .resident_nf4 import FrozenWeightCache
    if _resident_cache is None:
        _resident_cache = FrozenWeightCache(_decode, max_bytes=max_bytes)
    return _resident_cache


def _linear_weight(packed, state, dtype):
    # The forward fallback decodes directly to its compute dtype. Preserve
    # this distinction when QuantState.dtype differs from that dtype.
    if (_resident_cache is not None and dtype == torch.float16
            and getattr(state,'dtype',None) == torch.float16
            and isinstance(packed,torch.Tensor) and packed.is_cuda
            and packed.dtype == torch.uint8 and not packed.requires_grad):
        return _resident_cache.get(packed,state,dtype)
    return _decode(packed,state,dtype)


def _readonly_weight(packed, state, dtype):
    """Internal backward consumer: native state-dtype decode, then cast."""
    return _linear_weight(packed,state,state.dtype).to(dtype)


def _uses_native_custom_gemm(A, state):
    from bitsandbytes.backends.cuda import ops

    rows = A.numel() // A.shape[-1]
    if rows > ops._gemm_4bit_custom_max_m or A.shape[-1] % state.blocksize:
        return False
    return bool(ops._gemm_4bit_use_custom_fn(
        A.device.index, A.dtype, rows, state.shape[0], state.shape[1]))


def _native_forward(A, packed, bias, state):
    packed = packed.view(-1, 1)
    if state.nested:
        return torch.ops.bitsandbytes.gemm_4bit.default(
            A, packed, state.shape, state.state2.absmax, state.blocksize,
            state.quant_type, bias=bias, absmax_8bit=state.absmax,
            absmax_code=state.state2.code, absmax_offset=state.offset)
    return torch.ops.bitsandbytes.gemm_4bit.default(
        A, packed, state.shape, state.absmax, state.blocksize,
        state.quant_type, bias=bias)


def _linear_forward(A, packed, bias, state):
    """Shared non-autograd forward for MatMulNF4 and frozen MLP4bit."""
    if _uses_native_custom_gemm(A, state):
        return _native_forward(A, packed, bias, state)
    return torch.nn.functional.linear(A, _linear_weight(packed, state, A.dtype), bias)


class MatMulNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, packed, bias, state):
        out = _linear_forward(A, packed, bias, state)
        ctx.state = state
        ctx.save_for_backward(packed)
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        packed, = ctx.saved_tensors
        grad_A = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output,
                                  _readonly_weight(packed, ctx.state, grad_output.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        return grad_A, None, grad_bias, None


def make_matmul_4bit(original):
    @functools.wraps(original)
    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if (out is not None or not isinstance(A, torch.Tensor) or not A.is_cuda
                or A.numel() == 0 or A.dtype != torch.float16
                or not _packed_supported(B, quant_state)
                or quant_state.dtype != torch.float16
                or A.device != B.device or A.shape[-1] != quant_state.shape[1]):
            return original(A, B, quant_state, out=out, bias=bias)
        return MatMulNF4.apply(A, B, bias, quant_state)
    matmul_4bit._soup_team_nf4 = True
    return matmul_4bit


def patch_bnb_matmul_4bit():
    import bitsandbytes
    if not getattr(bitsandbytes.matmul_4bit, '_soup_team_nf4', False):
        bitsandbytes.matmul_4bit = make_matmul_4bit(bitsandbytes.matmul_4bit)


def warmup():
    from .warmup import load_compiled

    f32, u8 = torch.float32, torch.uint8
    load_compiled(nf4_dequant_kernel.warmup(u8, u8, f32, f32, f32, f32, torch.float16,
                              16 * 1536, 64, True, 256, BLOCK, grid=(1,),
                              num_warps=4, enable_fp_fusion=False))

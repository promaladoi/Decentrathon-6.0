"""Decode each NF4 byte once and reuse its scale for both stored weights.

The format and fallback are Vladimir's shared implementation. This team-written
kernel changes the element layout, not the quantized model or arithmetic.
No dense weight cache is retained. See CODEX_V17_EXPERIMENT.md.
"""
import math

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 1024
# The T4 microbenchmark found the byte-paired path slightly faster on small
# attention projections but slower on large MLP weights. Keep the old decoder
# on the latter; this is shape dispatch, not a change of quantized values.
PAIRED_MAX_ELEMENTS = 4_194_304


@triton.jit
def paired_nf4_kernel(PACKED, CODE, ABSMAX, ABSMAX2, CODE2, OFFSET, OUT, n,
                       QBLOCK: tl.constexpr, NESTED: tl.constexpr,
                       BLOCK_BYTES: tl.constexpr):
    byte_index = tl.program_id(0).to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    valid = byte_index < (n + 1) // 2
    packed = tl.load(PACKED + byte_index, valid, other=0).to(tl.int32)
    high = tl.load(CODE + (packed >> 4), valid, other=0.0)
    low = tl.load(CODE + (packed & 15), valid, other=0.0)
    qblock = byte_index // (QBLOCK // 2)
    if NESTED:
        encoded_scale = tl.load(ABSMAX + qblock, valid, other=0).to(tl.int32)
        scale = (tl.load(CODE2 + encoded_scale, valid, other=0.0)
                 * tl.load(ABSMAX2 + qblock // 256, valid, other=0.0)
                 + tl.load(OFFSET))
    else:
        scale = tl.load(ABSMAX + qblock, valid, other=0.0)
    # A byte stores its even weight in the high nibble. Join in a new minor
    # dimension before flattening: [high0, low0, high1, low1, ...]. Contiguous
    # output stores also handle an odd final weight without writing its padding.
    values = tl.reshape(tl.join(high * scale, low * scale), (2 * BLOCK_BYTES,))
    output_index = tl.program_id(0).to(tl.int64) * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
    tl.store(OUT + output_index, values.to(OUT.dtype.element_ty), output_index < n)


def dequantize_pairs(packed, state, dtype):
    output = torch.empty(tuple(state.shape), device=packed.device, dtype=dtype)
    n = output.numel()
    packed = packed.reshape(-1).contiguous()
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    code = state.code.to(device=packed.device, dtype=torch.float32).contiguous()
    if state.nested:
        nested = state.state2
        args = (state.absmax.contiguous(), nested.absmax.float().contiguous(),
                nested.code.float().contiguous(), state.offset.float().reshape(1))
    else:
        absmax = state.absmax.float().contiguous()
        args = (absmax, absmax, code, code)
    paired_nf4_kernel[(triton.cdiv(n, 2 * BLOCK_BYTES),)](
        packed, code, *args, output, n, state.blocksize, bool(state.nested),
        BLOCK_BYTES, num_warps=4)
    return output


def warmup():
    f32, u8 = torch.float32, torch.uint8
    paired_nf4_kernel.warmup(u8, f32, u8, f32, f32, f32, torch.float16,
                             16 * 1536, 64, True, BLOCK_BYTES,
                             grid=(1,), num_warps=4)
    from .team_vladimir.nf4 import nf4_dequant_kernel, BLOCK
    nf4_dequant_kernel.warmup(u8, f32, u8, f32, f32, f32, torch.float16,
                              16 * 1536, QBLOCK=64, NESTED=True, BLOCK=BLOCK,
                              grid=(1,), num_warps=8)


def patch_paired_nf4():
    from .team_vladimir import nf4, mlp, unpad
    original = nf4.dequantize_nf4
    if getattr(original, '_paired_nf4_original', None) is not None:
        return

    def decode(packed, state, dtype):
        if (packed.is_cuda and dtype in (torch.float16, torch.float32)
                and nf4.supported(state) and state.blocksize > 0
                and state.blocksize % 2 == 0
                and 0 < math.prod(state.shape) <= PAIRED_MAX_ELEMENTS):
            return dequantize_pairs(packed, state, dtype)
        return original(packed, state, dtype)

    decode._paired_nf4_original = original
    nf4.dequantize_nf4 = decode
    # These modules imported the decoder by name, so replace those references
    # as well. MatMulNF4 uses the nf4 module's global and needs no class change.
    mlp.dequantize_nf4 = decode
    unpad.dequantize_nf4 = decode
    nf4.warmup = warmup

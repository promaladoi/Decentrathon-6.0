"""Ephemeral FP32 nested-scale decoding for the inactive paired GEMM V2.

One launch reconstructs both weights' scale vectors, not their dense values.
The arrays are local to one call, never cached or persisted. Multiplication
and offset addition remain separate FP32 operations, as in nf4.py.
"""

import math

import torch
import triton
import triton.language as tl

from .nf4 import supported as _supported_state


@triton.jit(do_not_specialize=["COUNT"])
def nested_scale_pair_kernel(G_INDEX, G_CODE8, G_MAX, G_OFFSET,
                              U_INDEX, U_CODE8, U_MAX, U_OFFSET,
                              SCALES, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    gate_index = tl.load(G_INDEX + index, valid, other=0).to(tl.int32)
    gate_code = tl.load(G_CODE8 + gate_index).to(tl.float32)
    gate_max = tl.load(G_MAX + index // 256, valid, other=0.0).to(tl.float32)
    gate = gate_code * gate_max
    gate = gate + tl.load(G_OFFSET).to(tl.float32)
    up_index = tl.load(U_INDEX + index, valid, other=0).to(tl.int32)
    up_code = tl.load(U_CODE8 + up_index).to(tl.float32)
    up_max = tl.load(U_MAX + index // 256, valid, other=0.0).to(tl.float32)
    up = up_code * up_max
    up = up + tl.load(U_OFFSET).to(tl.float32)
    tl.store(SCALES + index, gate, valid)
    tl.store(SCALES + COUNT + index, up, valid)


def predecode_nested_scales_pair(gate, up):
    """Return two fresh scale vectors equal to the existing decoder's FP32 scale."""
    if (not _supported_state(gate) or not _supported_state(up)
            or not gate.nested or not up.nested
            or gate.blocksize != 64 or up.blocksize != 64
            or tuple(gate.shape) != tuple(up.shape) or gate.code.device != up.code.device):
        raise ValueError("scale pair requires matching CUDA nested NF4/64 states")
    count = triton.cdiv(math.prod(gate.shape), 64)
    scales = torch.empty((2, count), device=gate.code.device, dtype=torch.float32)
    nested_scale_pair_kernel[(triton.cdiv(count, 1024),)](
        gate.absmax, gate.state2.code, gate.state2.absmax, gate.offset,
        up.absmax, up.state2.code, up.state2.absmax, up.offset,
        scales, count, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return scales

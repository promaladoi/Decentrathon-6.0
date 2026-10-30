"""Inactive SM75 pair V2: ephemeral predecoded FP32 scales plus register NF4.

The separate per-weight FP32 accumulators, FP16 output boundaries, ordered
FP32 sum and strict oracle are unchanged. A first elementwise kernel decodes
only two scale vectors, then the Turing GEMM loads one scale per weight row.
There is no persistent cache or dense-weight/Split-K workspace. For K8960,
N1536 the scale workspace is 1,720,320 bytes and dies after the current call.
This may still lose to cuBLAS and is not installed into training.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .nf4 import _packed_supported
from .nf4_scale_pair import predecode_nested_scales_pair
from .sm75_gluon import _mma_turing


CONFIGS = {
    "m32n64k32w4s1_scales": (32, 64, 32, 4, 1),
    "m64n64k32w4s1_scales": (64, 64, 32, 4, 1),
    "m128n32k32w4s1_scales": (128, 32, 32, 4, 1),
}
DEFAULT_CONFIG = "m128n32k32w4s1_scales"


@gluon.jit
def _scaled_nf4_tile(PACKED, CODE4, SCALES, scale_offset: gl.constexpr,
                      kk, columns, K: gl.constexpr, N: gl.constexpr,
                      BLOCK_N: gl.constexpr):
    valid_row = kk < K
    output_quant_block = (gl.program_id(1) * BLOCK_N) // 64
    quant_block = kk.to(gl.int64) * (N // 64) + output_quant_block
    scale = gl.load(SCALES + scale_offset + quant_block, valid_row, other=0.0)
    position = kk[:, None].to(gl.int64) * N + columns[None, :]
    byte = gl.load(PACKED + position // 2, valid_row[:, None], other=0).to(gl.int32)
    nibble = gl.where((position & 1) == 0, byte >> 4, byte & 15)
    code = gl.load(CODE4 + nibble).to(gl.float32)
    value = code * scale[:, None]
    return gl.where(valid_row[:, None], value, 0.0).to(gl.float16)


@gluon.jit(do_not_specialize=["M"])
def sm75_mlp_pair_v2_kernel(DG, DU, G_PACKED, G_CODE4, U_PACKED, U_CODE4, SCALES,
                             OUT, M, N: gl.constexpr, K: gl.constexpr,
                             BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    warps_m: gl.constexpr = 4 if BLOCK_M >= 128 and BLOCK_N == 32 else 2
    warps_n: gl.constexpr = 1 if BLOCK_M >= 128 and BLOCK_N == 32 else 2
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 1], warps_per_cta=[warps_m, warps_n], instr_shape=[16, 8])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, load_layout))
    columns = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, load_layout))
    ka = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, load_layout))
    kb = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, load_layout))
    gate_accumulator = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, layout=mma_layout)
    up_accumulator = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, layout=mma_layout)
    for tile in range(gl.cdiv(K, BLOCK_K)):
        a_k, b_k = tile * BLOCK_K + ka, tile * BLOCK_K + kb
        source = rows[:, None].to(gl.int64) * K + a_k[None, :]
        mask = (rows[:, None] < M) & (a_k[None, :] < K)
        dg = gl.load(DG + source, mask, other=0.0)
        du = gl.load(DU + source, mask, other=0.0)
        gate = _scaled_nf4_tile(G_PACKED, G_CODE4, SCALES, 0,
                                 b_k, columns, K, N, BLOCK_N)
        up = _scaled_nf4_tile(U_PACKED, U_CODE4, SCALES, K * (N // 64),
                               b_k, columns, K, N, BLOCK_N)
        gate_accumulator = _mma_turing(gl.convert_layout(dg, a_layout),
                                        gl.convert_layout(gate, b_layout), gate_accumulator)
        up_accumulator = _mma_turing(gl.convert_layout(du, a_layout),
                                      gl.convert_layout(up, b_layout), up_accumulator)
    gate_half = gate_accumulator.to(gl.float16).to(gl.float32)
    up_half = up_accumulator.to(gl.float16).to(gl.float32)
    result = gl.convert_layout(gate_half + up_half, load_layout)
    destination = rows[:, None].to(gl.int64) * N + columns[None, :]
    gl.store(OUT + destination, result.to(OUT.dtype.element_ty), mask=rows[:, None] < M)


def _state(weight):
    state = getattr(weight, "quant_state", None)
    if (not _packed_supported(weight, state) or not state.nested
            or state.dtype != torch.float16 or state.blocksize != 64):
        return None
    return state


def supported(dg, du, gate_weight, up_weight, output_dtype=torch.float32):
    if (not isinstance(dg, torch.Tensor) or not isinstance(du, torch.Tensor)
            or not dg.is_cuda or dg.device != du.device
            or dg.dtype != torch.float16 or du.dtype != torch.float16
            or dg.ndim != 2 or du.shape != dg.shape
            or not dg.is_contiguous() or not du.is_contiguous()
            or output_dtype not in (torch.float16, torch.float32)
            or (torch.is_grad_enabled() and (dg.requires_grad or du.requires_grad))):
        return False
    gate, up = _state(gate_weight), _state(up_weight)
    return (gate is not None and up is not None
            and tuple(gate.shape) == tuple(up.shape) and gate.shape[0] == dg.shape[1]
            and gate.shape[1] % 64 == 0
            and gate_weight.device == up_weight.device == dg.device)


def sm75_mlp_grad_pair_v2(dg, du, gate_weight, up_weight, output_dtype=torch.float32,
                           config=DEFAULT_CONFIG, *, return_compiled=False):
    """Uninstalled explicit pair including fresh scale predecode every call."""
    if config not in CONFIGS:
        raise ValueError("unknown SM75 V2 paired-GEMM tile")
    if not supported(dg, du, gate_weight, up_weight, output_dtype):
        raise ValueError("SM75 V2 pair requires dense half gradients and matching nested NF4/64 weights with N%64=0")
    if triton.__version__ != "3.6.0" or torch.cuda.get_device_capability(dg.device) != (7, 5):
        raise RuntimeError("explicit Turing MMA layout is pinned to Triton 3.6.0 and SM75")
    gate, up = gate_weight.quant_state, up_weight.quant_state
    m, k = dg.shape
    n = gate.shape[1]
    output = torch.empty((m, n), device=dg.device, dtype=output_dtype)
    if m == 0:
        return (output, None) if return_compiled else output
    scales = predecode_nested_scales_pair(gate, up)
    bm, bn, bk, warps, stages = CONFIGS[config]
    compiled = sm75_mlp_pair_v2_kernel[(triton.cdiv(m, bm), n // bn)](
        dg, du, gate_weight, gate.code, up_weight, up.code, scales, output,
        m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return (output, compiled) if return_compiled else output

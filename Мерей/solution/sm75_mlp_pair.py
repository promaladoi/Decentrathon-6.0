"""Inactive Turing Tensor-Core paired NF4 MLP input-gradient experiment.

Uses the team's proven explicit Gluon NVMMADistributedLayout [2,1] path, not
ordinary tl.dot's deprecated pre-SM80 FMA fallback. Each product accumulates
independently in FP32 and is independently rounded to FP16 before the ordered
FP32 addition. Weights are decoded into register tiles; no dense weight cache,
split-K buffer or intermediate GEMM output is allocated. Nothing installs this
module into production. Strict accuracy and real T4 measurements decide use.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .nf4 import _packed_supported
from .sm75_gluon import _mma_turing


CONFIGS = {
    "m16n64k32w4s1": (16, 64, 32, 4, 1),
    "m32n64k32w4s1": (32, 64, 32, 4, 1),
    "m64n64k32w4s1": (64, 64, 32, 4, 1),
    "m16n64k64w4s1": (16, 64, 64, 4, 1),
    "m32n64k64w4s1": (32, 64, 64, 4, 1),
    "m64n64k64w4s1": (64, 64, 64, 4, 1),
}
DEFAULT_CONFIG = "m32n64k32w4s1"


@gluon.jit
def _nested_nf4_row_tile(PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                          kk, columns, K: gl.constexpr, N: gl.constexpr):
    # N and each 64-column output tile align with quantization blocks. The
    # same nested scale serves all 64 weights in this K row.
    valid_row = kk < K
    quant_block = kk.to(gl.int64) * (N // 64) + gl.program_id(1)
    scale_index = gl.load(ABSMAX + quant_block, valid_row, other=0).to(gl.int32)
    scale_code = gl.load(CODE8 + scale_index).to(gl.float32)
    scale_maximum = gl.load(ABSMAX2 + quant_block // 256, valid_row, other=0.0).to(gl.float32)
    scale = scale_code * scale_maximum
    scale = scale + gl.load(OFFSET).to(gl.float32)
    position = kk[:, None].to(gl.int64) * N + columns[None, :]
    byte = gl.load(PACKED + position // 2, valid_row[:, None], other=0).to(gl.int32)
    nibble = gl.where((position & 1) == 0, byte >> 4, byte & 15)
    code = gl.load(CODE4 + nibble).to(gl.float32)
    value = code * scale[:, None]
    return gl.where(valid_row[:, None], value, 0.0).to(gl.float16)


@gluon.jit(do_not_specialize=["M"])
def sm75_mlp_pair_kernel(DG, DU,
                          G_PACKED, G_ABSMAX, G_CODE4, G_ABSMAX2, G_CODE8, G_OFFSET,
                          U_PACKED, U_ABSMAX, U_CODE4, U_ABSMAX2, U_CODE8, U_OFFSET,
                          OUT, M, N: gl.constexpr, K: gl.constexpr,
                          BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr):
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    warps_m: gl.constexpr = 1 if BLOCK_M == 16 else 2
    warps_n: gl.constexpr = 4 if BLOCK_M == 16 else 2
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 1], warps_per_cta=[warps_m, warps_n], instr_shape=[16, 8])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, load_layout))
    columns = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    ka = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, load_layout))
    kb = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, load_layout))
    gate_accumulator = gl.full((BLOCK_M, 64), 0, gl.float32, layout=mma_layout)
    up_accumulator = gl.full((BLOCK_M, 64), 0, gl.float32, layout=mma_layout)
    for tile in range(gl.cdiv(K, BLOCK_K)):
        a_k, b_k = tile * BLOCK_K + ka, tile * BLOCK_K + kb
        input_position = rows[:, None].to(gl.int64) * K + a_k[None, :]
        input_mask = (rows[:, None] < M) & (a_k[None, :] < K)
        dg = gl.load(DG + input_position, input_mask, other=0.0)
        du = gl.load(DU + input_position, input_mask, other=0.0)
        gate_weight = _nested_nf4_row_tile(
            G_PACKED, G_ABSMAX, G_CODE4, G_ABSMAX2, G_CODE8, G_OFFSET, b_k, columns, K, N)
        up_weight = _nested_nf4_row_tile(
            U_PACKED, U_ABSMAX, U_CODE4, U_ABSMAX2, U_CODE8, U_OFFSET, b_k, columns, K, N)
        gate_accumulator = _mma_turing(gl.convert_layout(dg, a_layout),
                                        gl.convert_layout(gate_weight, b_layout), gate_accumulator)
        up_accumulator = _mma_turing(gl.convert_layout(du, a_layout),
                                      gl.convert_layout(up_weight, b_layout), up_accumulator)
    # These are two distinct cuBLAS-output FP16 boundaries, not one rounding
    # after adding FP32 accumulators. Preserve the cancellation contract.
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


def sm75_mlp_grad_pair(dg, du, gate_weight, up_weight, output_dtype=torch.float32,
                        config=DEFAULT_CONFIG, *, return_compiled=False):
    """Explicit, uninstalled SM75 pair; graph-building calls must use Torch."""
    if config not in CONFIGS:
        raise ValueError("unknown SM75 paired-GEMM tile")
    if not supported(dg, du, gate_weight, up_weight, output_dtype):
        raise ValueError("SM75 pair requires dense half gradients and matching nested NF4/64 weights with N%64=0")
    if triton.__version__ != "3.6.0" or torch.cuda.get_device_capability(dg.device) != (7, 5):
        raise RuntimeError("explicit Turing MMA layout is pinned to Triton 3.6.0 and SM75")
    gate, up = gate_weight.quant_state, up_weight.quant_state
    m, k = dg.shape
    n = gate.shape[1]
    output = torch.empty((m, n), device=dg.device, dtype=output_dtype)
    if m == 0:
        return (output, None) if return_compiled else output
    bm, _bn, bk, warps, stages = CONFIGS[config]
    compiled = sm75_mlp_pair_kernel[(triton.cdiv(m, bm), n // 64)](
        dg, du, gate_weight, gate.absmax, gate.code, gate.state2.absmax, gate.state2.code, gate.offset,
        up_weight, up.absmax, up.code, up.state2.absmax, up.state2.code, up.offset,
        output, m, n, k, BLOCK_M=bm, BLOCK_K=bk,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return (output, compiled) if return_compiled else output

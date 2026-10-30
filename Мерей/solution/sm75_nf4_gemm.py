"""Inactive nested-NF4 GEMM using the proved explicit Gluon SM75 MMA path.

The existing same-team solution/nf4_gemm.py supplies the decode/shape contract
and unchanged Torch reference. Its decode arithmetic is translated here to
Gluon (a legacy TL-jit function cannot be called from Gluon). The tile/layout
implementation is original; _mma_turing comes from our sm75_gluon proof.

Forward: A @ W.T. Input gradient: dY @ W. Nested scale multiplication and
offset addition are separate; decoded weights become FP16 before MMA, FP32
accumulators become FP16 at output. No dense weight cache or split-K storage.
No production apply()/dispatch imports this module. Standalone performance
and accuracy qualification do not establish whole-model quality or speed.
"""

from dataclasses import dataclass

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .nf4_gemm import _validate, reference_nf4_gemm
from .sm75_gluon import _mma_turing


@dataclass(frozen=True)
class Sm75Tile:
    m: int = 32
    n: int = 64
    k: int = 32
    warps: int = 4
    stages: int = 1


TILES = {
    'm32n64k32s1': Sm75Tile(),
    'm64n64k32s1': Sm75Tile(64, 64),
    'm32n128k32s1': Sm75Tile(32, 128),
    'm64n128k32s1': Sm75Tile(64, 128),
    'm32n64k32s2': Sm75Tile(stages=2),
    'm64n64k32s2': Sm75Tile(64, 64, stages=2),
}
DEFAULT_TILE = TILES['m32n64k32s1']


@gluon.jit
def _nested_row_scale(ABSMAX, ABSMAX2, CODE8, OFFSET, rows, col_start,
                       W_N: gl.constexpr, W_K: gl.constexpr):
    blocks = rows.to(gl.int64) * (W_K // 64) + col_start // 64
    valid = (rows < W_N) & (col_start < W_K)
    code = gl.load(ABSMAX + blocks, mask=valid, other=0).to(gl.int32)
    encoded = gl.load(CODE8 + code).to(gl.float32)
    maximum = gl.load(ABSMAX2 + blocks // 256, mask=valid, other=0.0).to(gl.float32)
    scale = encoded * maximum
    return scale + gl.load(OFFSET).to(gl.float32)


@gluon.jit
def _nf4_tile(PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
               rows, columns, col_start, W_N: gl.constexpr, W_K: gl.constexpr,
               COLS: gl.constexpr):
    # W_K and starts are quant-block aligned for a 64/128-wide output tile;
    # 32-wide reduction tiles occupy one half of an aligned quant block.
    valid = (rows[:, None] < W_N) & (columns[None, :] < W_K)
    positions = rows[:, None].to(gl.int64) * W_K + columns[None, :]
    byte = gl.load(PACKED + positions // 2, mask=valid, other=0).to(gl.int32)
    nibble = gl.where((positions & 1) == 0, byte >> 4, byte & 15)
    value = gl.load(CODE4 + nibble).to(gl.float32)
    left = _nested_row_scale(ABSMAX, ABSMAX2, CODE8, OFFSET, rows, col_start, W_N, W_K)
    if COLS <= 64:
        scale = left[:, None]
    else:
        gl.static_assert(COLS == 128, 'only one or two quant blocks per row are supported')
        right = _nested_row_scale(ABSMAX, ABSMAX2, CODE8, OFFSET,
                                  rows, col_start + 64, W_N, W_K)
        scale = gl.where(columns[None, :] - col_start < 64, left[:, None], right[:, None])
    decoded = value * scale
    # Invalid reduction lanes must be zero even for an exceptional scale.
    return gl.where(valid, decoded, 0.0).to(gl.float16)


@gluon.jit(do_not_specialize=['M'])
def sm75_nf4_gemm_kernel(A, PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET, OUT,
                          M, stride_am, stride_ak, W_N: gl.constexpr, W_K: gl.constexpr,
                          TRANSPOSE_WEIGHT: gl.constexpr,
                          BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    gl.static_assert(W_K % 64 == 0 and BK == 32, 'NF4 row-scale alignment contract')
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 1], warps_per_cta=[2, 2], instr_shape=[16, 8])
    lhs_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    width: gl.constexpr = W_N if TRANSPOSE_WEIGHT else W_K
    reduction: gl.constexpr = W_K if TRANSPOSE_WEIGHT else W_N
    rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, load_layout))
    columns = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, load_layout))
    a_inner = gl.arange(0, BK, layout=gl.SliceLayout(0, load_layout))
    weight_rows = gl.arange(0, BK, layout=gl.SliceLayout(1, load_layout))
    forward_rows = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, load_layout))
    accumulator = gl.full((BM, BN), 0, gl.float32, layout=mma_layout)
    for iteration in range(gl.cdiv(reduction, BK)):
        ka = iteration * BK + a_inner
        left = gl.load(A + rows[:, None].to(gl.int64) * stride_am + ka[None, :] * stride_ak,
                       mask=(rows[:, None] < M) & (ka[None, :] < reduction), other=0.0)
        if TRANSPOSE_WEIGHT:
            weight = _nf4_tile(PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                               forward_rows, ka, iteration * BK, W_N, W_K, BK)
            right = gl.permute(weight, (1, 0))
        else:
            kr = iteration * BK + weight_rows
            right = _nf4_tile(PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                              kr, columns, gl.program_id(1) * BN, W_N, W_K, BN)
        left = gl.convert_layout(left, lhs_layout)
        right = gl.convert_layout(right, rhs_layout)
        accumulator = _mma_turing(left, right, accumulator)
    result = gl.convert_layout(accumulator, load_layout).to(gl.float16)
    gl.store(OUT + rows[:, None].to(gl.int64) * width + columns[None, :], result,
             mask=(rows[:, None] < M) & (columns[None, :] < width))


def _tile_supported(tile):
    return (isinstance(tile, Sm75Tile) and tile.m in (32, 64) and tile.n in (64, 128)
            and tile.k == 32 and tile.warps == 4 and tile.stages in (1, 2))


def sm75_nf4_gemm(a, packed, state, *, transpose_weight=True, tile=DEFAULT_TILE,
                   return_compiled=False):
    """Explicit experiment. Higher-order graph requests retain the old Torch path."""
    if not _tile_supported(tile):
        raise ValueError('unsupported SM75 Gluon NF4 tile')
    _validate(a, packed, state, transpose_weight)
    if (not state.nested or state.dtype != torch.float16 or state.blocksize != 64
            or state.shape[1] % 64 or triton.__version__ != '3.6.0'
            or torch.cuda.get_device_capability(a.device) != (7, 5)):
        raise ValueError('SM75 Gluon requires Triton 3.6, nested NF4/64 half weights with aligned rows')
    if torch.is_grad_enabled() and a.requires_grad:
        result = reference_nf4_gemm(a, packed, state, transpose_weight=transpose_weight)
        return (result, None) if return_compiled else result
    width = state.shape[0 if transpose_weight else 1]
    output = torch.empty((a.shape[0], width), device=a.device, dtype=torch.float16)
    if a.shape[0] == 0:
        return (output, None) if return_compiled else output
    compiled = sm75_nf4_gemm_kernel[(triton.cdiv(a.shape[0], tile.m), triton.cdiv(width, tile.n))](
        a, packed, state.absmax, state.code, state.state2.absmax, state.state2.code, state.offset,
        output, a.shape[0], a.stride(0), a.stride(1), W_N=state.shape[0], W_K=state.shape[1],
        TRANSPOSE_WEIGHT=transpose_weight, BM=tile.m, BN=tile.n, BK=tile.k,
        num_warps=tile.warps, num_stages=tile.stages, enable_fp_fusion=False)
    return (output, compiled) if return_compiled else output

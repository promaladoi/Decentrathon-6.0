"""Experimental in-register NF4 decode plus FP16 Tensor-Core GEMM.

This module is not installed into training. Each output tile decodes its own
weight tile and accumulates in FP32; no dense weight cache or split-K buffer is
allocated. Forward computes A @ W.T; the input-gradient direction computes
dY @ W. A decoded weight is rounded to FP16 before the dot product, matching
the task's existing decoder/cast boundary. Nested scale multiplication and
offset addition remain separate (enable_fp_fusion=False).

This is an original tile implementation using Triton's documented dot/load
primitives: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
It is an experiment until accuracy, cold compilation, latency and whole-run
peak memory are measured on T4. A faster kernel microbenchmark alone is not a
claim about full training speed or quality.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .nf4 import _decode, _packed_supported, dequantize_nf4


@dataclass(frozen=True)
class GemmTile:
    m: int = 32
    n: int = 64
    k: int = 64
    warps: int = 4
    stages: int = 2


DEFAULT_TILE = GemmTile()
BENCHMARK_TILES = (
    GemmTile(16, 64, 64),
    GemmTile(32, 64, 64),
    GemmTile(64, 64, 64),
    GemmTile(32, 128, 32),
    GemmTile(64, 128, 32),
    GemmTile(128, 64, 32, 8),
)


@triton.jit
def _nf4_scales(ABSMAX, ABSMAX2, CODE8, OFFSET, blocks, valid,
                 NESTED: tl.constexpr, SCALE_BLOCK: tl.constexpr):
    if NESTED:
        code = tl.load(ABSMAX + blocks, valid, other=0).to(tl.int32)
        encoded = tl.load(CODE8 + code).to(tl.float32)
        maximum = tl.load(ABSMAX2 + blocks // SCALE_BLOCK, valid, other=0.0).to(tl.float32)
        scale = encoded * maximum
        scale = scale + tl.load(OFFSET).to(tl.float32)
    else:
        scale = tl.load(ABSMAX + blocks, valid, other=0.0).to(tl.float32)
    return scale


@triton.jit
def _nf4_weight_tile(PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                      rows, cols, col_start, W_N: tl.constexpr, W_K: tl.constexpr,
                      QBLOCK: tl.constexpr, NESTED: tl.constexpr,
                      STATE_HALF: tl.constexpr, COLS: tl.constexpr):
    valid_row = rows < W_N
    valid = valid_row[:, None] & (cols[None, :] < W_K)
    flat = rows[:, None].to(tl.int64) * W_K + cols[None, :]
    byte = tl.load(PACKED + flat // 2, valid, other=0).to(tl.int32)
    nibble = tl.where(flat % 2 == 0, byte >> 4, byte & 15)
    code = tl.load(CODE4 + nibble).to(tl.float32)
    if W_K % QBLOCK == 0 and COLS <= QBLOCK and QBLOCK % COLS == 0:
        # The Qwen shapes and 32/64-wide tiles stay inside one quant block.
        # Load its scale once per weight row and broadcast over the columns.
        blocks = rows.to(tl.int64) * (W_K // QBLOCK) + col_start // QBLOCK
        scale = _nf4_scales(ABSMAX, ABSMAX2, CODE8, OFFSET, blocks, valid_row,
                            NESTED, 256)
        value = code * scale[:, None]
    else:
        scale = _nf4_scales(ABSMAX, ABSMAX2, CODE8, OFFSET, flat // QBLOCK, valid,
                            NESTED, 256)
        value = code * scale
    if STATE_HALF:
        # The native backward decoder first rounds to QuantState.dtype.
        value = value.to(tl.float16).to(tl.float32)
    # The dot inputs are FP16 in this experiment. Mask before dot: invalid
    # K padding must be zero even if an exceptional scale/code is non-finite.
    return tl.where(valid, value, 0.0).to(tl.float16)


@triton.jit(do_not_specialize=["M"])
def nf4_gemm_kernel(A, PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET, OUT,
                     M, stride_am, stride_ak,
                     W_N: tl.constexpr, W_K: tl.constexpr,
                     QBLOCK: tl.constexpr, NESTED: tl.constexpr,
                     STATE_HALF: tl.constexpr, TRANSPOSE_WEIGHT: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    output_width: tl.constexpr = W_N if TRANSPOSE_WEIGHT else W_K
    reduction: tl.constexpr = W_K if TRANSPOSE_WEIGHT else W_N
    out_rows = tl.program_id(0) * BM + tl.arange(0, BM)
    out_cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, tl.cdiv(reduction, BK)):
        k = start * BK + inner
        left = tl.load(A + out_rows[:, None] * stride_am + k[None, :] * stride_ak,
                       (out_rows[:, None] < M) & (k[None, :] < reduction), other=0.0)
        if TRANSPOSE_WEIGHT:
            weight = _nf4_weight_tile(
                PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                out_cols, k, start * BK, W_N, W_K, QBLOCK, NESTED, STATE_HALF, BK)
            right = tl.trans(weight)
        else:
            right = _nf4_weight_tile(
                PACKED, ABSMAX, CODE4, ABSMAX2, CODE8, OFFSET,
                k, out_cols, tl.program_id(1) * BN, W_N, W_K,
                QBLOCK, NESTED, STATE_HALF, BN)
        accumulator = tl.dot(left, right, accumulator, input_precision="ieee")
    tl.store(OUT + out_rows[:, None] * output_width + out_cols[None, :],
              accumulator.to(tl.float16),
              (out_rows[:, None] < M) & (out_cols[None, :] < output_width))


def _validate_tile(tile):
    if (not isinstance(tile, GemmTile) or tile.m not in (16, 32, 64, 128)
            or tile.n not in (16, 32, 64, 128) or tile.k not in (32, 64, 128)
            or tile.warps not in (4, 8) or tile.stages not in (1, 2, 3)):
        raise ValueError("unsupported experimental NF4 GEMM tile")


def _validate(a, packed, state, transpose_weight):
    if (not isinstance(a, torch.Tensor) or not a.is_cuda or a.dtype != torch.float16
            or a.ndim != 2 or not _packed_supported(packed, state)
            or a.device != packed.device or type(transpose_weight) is not bool
            or a.shape[1] != state.shape[1 if transpose_weight else 0]):
        raise ValueError("experimental NF4 GEMM requires a matching 2D CUDA FP16 input")


def reference_nf4_gemm(a, packed, state, *, transpose_weight=True):
    """The existing decoder plus torch.mm, without a persistent dense weight."""
    dense = (_decode(packed, state, a.dtype) if transpose_weight
             else dequantize_nf4(packed, state, a.dtype))
    return torch.mm(a, dense.t() if transpose_weight else dense)


def fused_nf4_gemm(a, packed, state, *, transpose_weight=True, tile=DEFAULT_TILE):
    """Explicit experimental GEMM; ordinary autograd calls retain torch's graph."""
    _validate_tile(tile)
    _validate(a, packed, state, transpose_weight)
    if torch.is_grad_enabled() and a.requires_grad:
        return reference_nf4_gemm(a, packed, state, transpose_weight=transpose_weight)
    output_width = state.shape[0 if transpose_weight else 1]
    output = torch.empty((a.shape[0], output_width), device=a.device, dtype=torch.float16)
    if a.shape[0] == 0:
        return output
    nested = bool(state.nested)
    if nested:
        second_max, code8, offset = state.state2.absmax, state.state2.code, state.offset
    else:
        second_max = code8 = offset = state.absmax
    nf4_gemm_kernel[(triton.cdiv(a.shape[0], tile.m), triton.cdiv(output_width, tile.n))](
        a, packed, state.absmax, state.code, second_max, code8, offset, output,
        a.shape[0], a.stride(0), a.stride(1),
        W_N=state.shape[0], W_K=state.shape[1], QBLOCK=state.blocksize,
        NESTED=nested, STATE_HALF=not transpose_weight and state.dtype == torch.float16,
        TRANSPOSE_WEIGHT=transpose_weight, BM=tile.m, BN=tile.n, BK=tile.k,
        num_warps=tile.warps, num_stages=tile.stages, enable_fp_fusion=False)
    return output


class _FusedNF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, packed, state, forward_tile, backward_tile):
        ctx.state, ctx.backward_tile = state, backward_tile
        tensors = [packed, state.absmax, state.code]
        if state.nested:
            tensors.extend((state.state2.absmax, state.state2.code, state.offset))
        ctx.save_for_backward(*tensors)
        return fused_nf4_gemm(a, packed, state, tile=forward_tile)

    @staticmethod
    def backward(ctx, grad):
        packed, *_state_tensors = ctx.saved_tensors
        # When create_graph=True the raw launcher selects the ordinary torch
        # expression, preserving differentiation through the upstream gradient.
        dx = fused_nf4_gemm(grad, packed, ctx.state, transpose_weight=False,
                            tile=ctx.backward_tile)
        return dx, None, None, None, None


def fused_nf4_linear(a, packed, state, *, forward_tile=DEFAULT_TILE, backward_tile=DEFAULT_TILE):
    """Opt-in bias-free frozen linear; not wired into Soup or bitsandbytes."""
    _validate_tile(forward_tile)
    _validate_tile(backward_tile)
    _validate(a, packed, state, True)
    return _FusedNF4Linear.apply(a, packed, state, forward_tile, backward_tile)

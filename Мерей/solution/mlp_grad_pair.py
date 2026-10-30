"""Inactive experiment: two NF4 MLP input-gradient products in one CTA.

Each product has its own FP32 tensor-core accumulator, then its own FP16
rounding, before the ordered FP32 addition and the caller's output cast.
Weights are decoded from nested NF4 directly into a reduction tile. No dense
weight cache, split-K workspace or intermediate GEMM outputs are allocated.

The quantization contract is the existing team's solution/nf4.py. This is an
original implementation, not a copy of another team's research kernel. The
same-team CODEX_V29_RESEARCH report at abc3dc63 records failed numeric oracles
for a different dense-weight pair kernel; it is a warning, not evidence that
this experiment matches cuBLAS. Full T4 accuracy/performance qualification is
required. No production apply()/dispatch imports this module.
"""

import torch
import triton
import triton.language as tl

from .nf4 import _packed_supported


# (M tile, output-N tile, reduction-K tile, warps, pipeline stages).
# No autotuner runs in training: the standalone diagnostic selects explicitly.
CONFIGS = {
    "m16n32k32w4s2": (16, 32, 32, 4, 2),
    "m32n32k32w4s2": (32, 32, 32, 4, 2),
    "m32n64k32w4s2": (32, 64, 32, 4, 2),
    "m32n64k32w4s3": (32, 64, 32, 4, 3),
    "m32n32k64w4s2": (32, 32, 64, 4, 2),
    "m32n64k32w8s2": (32, 64, 32, 8, 2),
    "m16n64k128w4s1_row_scale": (16, 64, 128, 4, 1),
}
DEFAULT_CONFIG = "m32n32k32w4s2"
ROW_SCALE_CONFIG = "m16n64k128w4s1_row_scale"


@triton.jit
def _nested_nf4_tile(PACKED, SCALE_INDEX, CODE4, SCALE_MAX, CODE8, OFFSET,
                     positions, valid):
    byte = tl.load(PACKED + positions // 2, mask=valid, other=0).to(tl.int32)
    nibble = tl.where((positions & 1) == 0, byte >> 4, byte & 15)
    code = tl.load(CODE4 + nibble).to(tl.float32)
    quant_block = positions // 64
    scale_index = tl.load(SCALE_INDEX + quant_block, mask=valid, other=0).to(tl.int32)
    scale_code = tl.load(CODE8 + scale_index).to(tl.float32)
    scale_max = tl.load(SCALE_MAX + quant_block // 256, mask=valid, other=0).to(tl.float32)
    scale = scale_code * scale_max
    scale = scale + tl.load(OFFSET).to(tl.float32)
    decoded = code * scale
    # Explicitly zero padded reduction lanes, including exceptional offsets.
    return tl.where(valid, decoded, 0.0).to(tl.float16)


@triton.jit(do_not_specialize=["M"])
def mlp_grad_pair_kernel(DG, DU,
                         G_PACKED, G_SCALE_INDEX, G_CODE4, G_SCALE_MAX, G_CODE8, G_OFFSET,
                         U_PACKED, U_SCALE_INDEX, U_CODE4, U_SCALE_MAX, U_CODE8, U_OFFSET,
                         OUT, M, N: tl.constexpr, K: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BLOCK_K)):
        kk = tile * BLOCK_K + reduction
        input_position = rows[:, None].to(tl.int64) * K + kk[None, :]
        input_valid = (rows[:, None] < M) & (kk[None, :] < K)
        dg = tl.load(DG + input_position, mask=input_valid, other=0.0)
        du = tl.load(DU + input_position, mask=input_valid, other=0.0)
        weight_position = kk[:, None].to(tl.int64) * N + columns[None, :]
        weight_valid = (kk[:, None] < K) & (columns[None, :] < N)
        gate = _nested_nf4_tile(G_PACKED, G_SCALE_INDEX, G_CODE4, G_SCALE_MAX,
                               G_CODE8, G_OFFSET, weight_position, weight_valid)
        up = _nested_nf4_tile(U_PACKED, U_SCALE_INDEX, U_CODE4, U_SCALE_MAX,
                             U_CODE8, U_OFFSET, weight_position, weight_valid)
        gate_acc = tl.dot(dg, gate, gate_acc)
        up_acc = tl.dot(du, up, up_acc)
    # Do not combine accumulators before the two native-output half boundaries.
    gate_half = gate_acc.to(tl.float16).to(tl.float32)
    up_half = up_acc.to(tl.float16).to(tl.float32)
    result = gate_half + up_half
    destination = rows[:, None].to(tl.int64) * N + columns[None, :]
    tl.store(OUT + destination, result.to(OUT.dtype.element_ty),
              mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def _row_scale_nf4_tile(PACKED, SCALE_INDEX, CODE4, SCALE_MAX, CODE8, OFFSET,
                        kk, K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr):
    # N is divisible by 64 and the output CTA covers exactly one NF4 block
    # per K row. Decode the scale chain once per row, not once per weight.
    output_block = tl.program_id(1)
    quant_block = kk * (N // 64) + output_block
    valid_row = kk < K
    scale_index = tl.load(SCALE_INDEX + quant_block, mask=valid_row, other=0).to(tl.int32)
    scale_code = tl.load(CODE8 + scale_index).to(tl.float32)
    scale_max = tl.load(SCALE_MAX + quant_block // 256, mask=valid_row, other=0).to(tl.float32)
    scale = scale_code * scale_max
    scale = scale + tl.load(OFFSET).to(tl.float32)
    # One byte load supplies both adjacent nibbles; CODE4 remains the actual
    # runtime codebook. No hardcoded assumption about its sixteen values.
    byte_column = tl.arange(0, 32)
    byte_position = kk[:, None].to(tl.int64) * (N // 2) + output_block * 32 + byte_column[None, :]
    byte = tl.load(PACKED + byte_position, mask=valid_row[:, None], other=0).to(tl.int32)
    high = tl.load(CODE4 + (byte >> 4)).to(tl.float32)
    low = tl.load(CODE4 + (byte & 15)).to(tl.float32)
    high = tl.where(valid_row[:, None], high * scale[:, None], 0.0).to(tl.float16)
    low = tl.where(valid_row[:, None], low * scale[:, None], 0.0).to(tl.float16)
    return tl.reshape(tl.join(high, low), (BLOCK_K, 64))


@triton.jit(do_not_specialize=["M"])
def mlp_grad_pair_row_scale_kernel(DG, DU,
                                   G_PACKED, G_SCALE_INDEX, G_CODE4, G_SCALE_MAX, G_CODE8, G_OFFSET,
                                   U_PACKED, U_SCALE_INDEX, U_CODE4, U_SCALE_MAX, U_CODE8, U_OFFSET,
                                   OUT, M, N: tl.constexpr, K: tl.constexpr,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    tl.static_assert(BLOCK_N == 64, "row-scale output tiles must match an NF4 block")
    tl.static_assert(N % 64 == 0, "row-scale rows must start at NF4 block boundaries")
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * 64 + tl.arange(0, 64)
    reduction = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BLOCK_K)):
        kk = tile * BLOCK_K + reduction
        position = rows[:, None].to(tl.int64) * K + kk[None, :]
        valid = (rows[:, None] < M) & (kk[None, :] < K)
        dg = tl.load(DG + position, mask=valid, other=0.0)
        du = tl.load(DU + position, mask=valid, other=0.0)
        gate = _row_scale_nf4_tile(G_PACKED, G_SCALE_INDEX, G_CODE4, G_SCALE_MAX,
                                  G_CODE8, G_OFFSET, kk, K, N, BLOCK_K)
        up = _row_scale_nf4_tile(U_PACKED, U_SCALE_INDEX, U_CODE4, U_SCALE_MAX,
                                U_CODE8, U_OFFSET, kk, K, N, BLOCK_K)
        gate_acc = tl.dot(dg, gate, gate_acc)
        up_acc = tl.dot(du, up, up_acc)
    result = gate_acc.to(tl.float16).to(tl.float32) + up_acc.to(tl.float16).to(tl.float32)
    destination = rows[:, None].to(tl.int64) * N + columns[None, :]
    tl.store(OUT + destination, result.to(OUT.dtype.element_ty), mask=rows[:, None] < M)


def _state(weight):
    state = getattr(weight, "quant_state", None)
    if (not _packed_supported(weight, state) or not state.nested
            or state.dtype != torch.float16 or state.blocksize != 64):
        return None
    return state


def supported(dg, du, gate_weight, up_weight, output_dtype=torch.float32):
    """Tensor/format eligibility only; this does not assert numerical admission."""
    if (not isinstance(dg, torch.Tensor) or not isinstance(du, torch.Tensor)
            or not dg.is_cuda or du.device != dg.device
            or dg.dtype != torch.float16 or du.dtype != torch.float16
            or dg.ndim != 2 or du.shape != dg.shape
            or not dg.is_contiguous() or not du.is_contiguous()
            or output_dtype not in (torch.float16, torch.float32)
            or (torch.is_grad_enabled() and (dg.requires_grad or du.requires_grad))):
        return False
    gate, up = _state(gate_weight), _state(up_weight)
    return (gate is not None and up is not None
            and tuple(gate.shape) == tuple(up.shape)
            and gate.shape[0] == dg.shape[1]
            and gate_weight.device == up_weight.device == dg.device)


def mlp_grad_pair(dg, du, gate_weight, up_weight, output_dtype=torch.float32,
                  config=DEFAULT_CONFIG):
    """Experimental ``(dg @ Wgate).half().float() + (du @ Wup).half().float()``.

    Target task: M real tokens, N=1536, K=8960, nested NF4/64 frozen weights.
    Smaller/odd dimensions are accepted for focused boundary qualification.
    This raw operation does not register autograd; graph-building inputs are
    rejected. Callers must retain their normal Torch path for higher orders.
    """
    if config not in CONFIGS:
        raise ValueError("unknown experimental MLP pair tile configuration")
    if not supported(dg, du, gate_weight, up_weight, output_dtype):
        raise ValueError("MLP pair requires matching contiguous CUDA half gradients and frozen nested NF4/64")
    gate, up = gate_weight.quant_state, up_weight.quant_state
    m, k = dg.shape
    n = gate.shape[1]
    if config == ROW_SCALE_CONFIG and n % 64:
        raise ValueError("row-scale MLP pair requires output width divisible by NF4 block 64")
    output = torch.empty((m, n), device=dg.device, dtype=output_dtype)
    if m == 0:
        return output
    bm, bn, bk, warps, stages = CONFIGS[config]
    kernel = mlp_grad_pair_row_scale_kernel if config == ROW_SCALE_CONFIG else mlp_grad_pair_kernel
    kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dg, du, gate_weight, gate.absmax, gate.code, gate.state2.absmax, gate.state2.code, gate.offset,
        up_weight, up.absmax, up.code, up.state2.absmax, up.state2.code, up.offset,
        output, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return output

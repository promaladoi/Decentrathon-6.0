"""Inactive SM75 hardware-path proof using explicit Gluon MMA layout.

Triton 3.6's ordinary AccelerateMatmul pass declines pre-SM80 targets. The
Gluon pipeline accepts explicit TTGIR layouts and bypasses that pass. Turing
is represented by NVMMADistributedLayout(version=[2, 1]), not Ampere [2, 0].
This original dense proof is not a ready-made external kernel, NF4 dispatch,
or a production patch. The small builtin calls the existing compiler's dot
semantic API because its public ampere.mma_v2 wrapper asserts version 2.0.
No installed library/compiler code is modified.

Primary pinned sources, read before implementation:
https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L320
https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/IR/Dialect.cpp#L2382
https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py#L320
https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp#L64
https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L437
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin


@builtin
def _mma_turing(a, b, accumulator, _semantic=None):
    """Emit an already-laid-out SM75 dot; no layout-selection/compiler patch."""
    layout = accumulator.type.layout
    assert isinstance(layout, gl.NVMMADistributedLayout) and layout.version == [2, 1]
    assert a.dtype == b.dtype == gl.float16 and accumulator.dtype == gl.float32
    assert isinstance(a.type.layout, gl.DotOperandLayout) and a.type.layout.operand_index == 0
    assert isinstance(b.type.layout, gl.DotOperandLayout) and b.type.layout.operand_index == 1
    assert a.type.layout.parent == b.type.layout.parent == layout
    value = _semantic.dot(a, b, accumulator, input_precision='ieee',
                          max_num_imprecise_acc=None, out_dtype=accumulator.dtype)
    return gl.tensor(value.handle, accumulator.type)


@gluon.jit(do_not_specialize=['M', 'N', 'K'])
def sm75_dense_mma_kernel(A, B, OUT, M, N, K,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 1], warps_per_cta=[2, 2], instr_shape=[16, 8])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, load_layout))
    columns = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, load_layout))
    ka = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, load_layout))
    kb = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, load_layout))
    accumulator = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, layout=mma_layout)
    for tile in range(gl.cdiv(K, BLOCK_K)):
        a_k, b_k = tile * BLOCK_K + ka, tile * BLOCK_K + kb
        a = gl.load(A + rows[:, None].to(gl.int64) * K + a_k[None, :],
                    mask=(rows[:, None] < M) & (a_k[None, :] < K), other=0.0)
        b = gl.load(B + b_k[:, None].to(gl.int64) * N + columns[None, :],
                    mask=(b_k[:, None] < K) & (columns[None, :] < N), other=0.0)
        a = gl.convert_layout(a, a_layout)
        b = gl.convert_layout(b, b_layout)
        accumulator = _mma_turing(a, b, accumulator)
    result = gl.convert_layout(accumulator, load_layout)
    gl.store(OUT + rows[:, None].to(gl.int64) * N + columns[None, :], result,
             mask=(rows[:, None] < M) & (columns[None, :] < N))


def sm75_dense_matmul(a, b, *, return_compiled=False):
    """Tiny proof API: dense half operands, FP32 result, no registered autograd."""
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or not a.is_cuda or a.device != b.device
            or a.dtype != torch.float16 or b.dtype != torch.float16
            or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
            or not a.is_contiguous() or not b.is_contiguous()
            or min(*a.shape, *b.shape) <= 0
            or (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad))):
        raise ValueError('SM75 Gluon proof requires nonempty contiguous CUDA half matrices without autograd')
    if triton.__version__ != '3.6.0' or torch.cuda.get_device_capability(a.device) != (7, 5):
        raise RuntimeError('This explicit-layout proof is pinned to Triton 3.6.0 and SM75')
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    compiled = sm75_dense_mma_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        a, b, output, m, n, k, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return (output, compiled) if return_compiled else output

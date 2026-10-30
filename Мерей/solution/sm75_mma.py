"""Minimal SM75 Tensor-Core proof using physical-lane inline PTX.

This is not installed into training and is not a tuned GEMM. It proves one
16x8x8 FP16 MMA with FP32 C/D without calling tl.dot or AccelerateMatmul.
All fragment addressing, loads, packing and stores live in one side-effecting
scalar asm block. With num_warps=1 that block runs once on every physical lane;
there is no assumption about a Triton logical tensor's blocked layout.

Lane fragments follow NVIDIA PTX ISA section 9.7.16.5.7 (mma.m16n8k8):
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
FP16 m16n8k8 is supported on SM75+. Scalar inline-asm lowering was inspected in
Triton v3.6.0 ElementwiseOpToLLVM.cpp, lines 258-303 and 332-399; it emits the
side-effecting LLVM inline-asm operation without a per-lane predicate.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sm75_mma_kernel(A, B, C, D):
    # One scalar invocation per physical thread. Global loads/stores are
    # inside PTX, so compiler-selected arange/distribution layouts are absent.
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u32 lane, group, thread, pair, ai, bi;
            .reg .u64 abase, bbase, cbase, dbase, ap, bp, cp, dp, offset;
            .reg .b16 a0, a1, a2, a3, b0, b1;
            .reg .b32 ar0, ar1, br0;
            .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;

            cvta.to.global.u64 abase, $1;
            cvta.to.global.u64 bbase, $2;
            cvta.to.global.u64 cbase, $3;
            cvta.to.global.u64 dbase, $4;
            mov.u32 lane, %laneid;
            shr.u32 group, lane, 2;
            and.b32 thread, lane, 3;
            shl.b32 pair, thread, 1;

            // A fragments: [group, 2*thread + {0,1}] and [group+8, ...].
            mad.lo.u32 ai, group, 8, pair;
            mul.wide.u32 offset, ai, 2;
            add.u64 ap, abase, offset;
            ld.global.b16 a0, [ap];
            ld.global.b16 a1, [ap+2];
            ld.global.b16 a2, [ap+128];
            ld.global.b16 a3, [ap+130];
            mov.b32 ar0, {a0, a1};
            mov.b32 ar1, {a2, a3};

            // B fragments: [2*thread + {0,1}, group].
            mad.lo.u32 bi, pair, 8, group;
            mul.wide.u32 offset, bi, 2;
            add.u64 bp, bbase, offset;
            ld.global.b16 b0, [bp];
            ld.global.b16 b1, [bp+16];
            mov.b32 br0, {b0, b1};

            // C/D fragments use the same row/column map as A, now FP32.
            mul.wide.u32 offset, ai, 4;
            add.u64 cp, cbase, offset;
            add.u64 dp, dbase, offset;
            ld.global.f32 c0, [cp];
            ld.global.f32 c1, [cp+4];
            ld.global.f32 c2, [cp+256];
            ld.global.f32 c3, [cp+260];

            mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
                {d0, d1, d2, d3}, {ar0, ar1}, {br0}, {c0, c1, c2, c3};

            st.global.f32 [dp], d0;
            st.global.f32 [dp+4], d1;
            st.global.f32 [dp+256], d2;
            st.global.f32 [dp+260], d3;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,l,l,~{memory}",
        # The elementwise frontend broadcasts all arguments together. Mixed
        # pointer element types must first become uniform 64-bit addresses.
        args=[A.to(tl.uint64), B.to(tl.uint64), C.to(tl.uint64), D.to(tl.uint64)],
        dtype=tl.int32, is_pure=False, pack=1)


def mma_16x8x8(a, b, c=None, *, out=None):
    """Explicit non-autograd proof of FP32(A_fp16 @ B_fp16) + C_fp32."""
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or not a.is_cuda or not b.is_cuda or a.device != b.device
            or a.dtype != torch.float16 or b.dtype != torch.float16
            or a.shape != (16, 8) or b.shape != (8, 8)
            or not a.is_contiguous() or not b.is_contiguous()):
        raise ValueError("proof requires contiguous CUDA FP16 A[16,8], B[8,8]")
    if torch.cuda.get_device_capability(a.device) != (7, 5):
        raise ValueError("this qualification proof targets SM75")
    if c is None:
        c = torch.zeros((16, 8), device=a.device, dtype=torch.float32)
    if (not isinstance(c, torch.Tensor) or c.device != a.device
            or c.dtype != torch.float32 or c.shape != (16, 8) or not c.is_contiguous()):
        raise ValueError("proof requires contiguous CUDA FP32 C[16,8]")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (a, b, c)):
        raise ValueError("the isolated MMA proof is not an autograd operation")
    if out is None:
        out = torch.empty((16, 8), device=a.device, dtype=torch.float32)
    if (not isinstance(out, torch.Tensor) or out.device != a.device
            or out.dtype != torch.float32 or out.shape != (16, 8) or not out.is_contiguous()):
        raise ValueError("proof output must be contiguous CUDA FP32 [16,8]")
    sm75_mma_kernel[(1,)](a, b, c, out, num_warps=1)
    return out

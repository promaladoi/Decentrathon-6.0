"""v52 inactive SM75 research: explicit FP16 MMA with FP32 accumulation.

Original team implementation of the documented NVIDIA PTX m16n8k8 fragment
mapping; no external matmul/kernel library or tl.dot lowering is used.
Four warps compute a 64x32 output tile, sharing A fragments across four N tiles.
This simple global-load prototype first establishes mapping/numerical evidence;
it is not installed in solution.apply and is not a claimed training speedup.

ISA reference: NVIDIA CUDA 11.0 PTX, section 9.7.13.4.7:
https://docs.nvidia.com/cuda/archive/11.0_GA/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pair(a, b):
    return a.to(tl.uint16, bitcast=True).to(tl.uint32) | (b.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)


@triton.jit
def _mma(a0, a1, b, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 '
        '{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};',
        constraints='=f,=f,=f,=f,r,r,r,f,f,f,f',
        args=[a0, a1, b, c0, c1, c2, c3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True, pack=1)


@triton.jit
def _load_pair(P, base, mask0, mask1, ALIGNED: tl.constexpr):
    if ALIGNED:
        # Even row strides and aligned base pointers make this exactly the
        # same two FP16 bit patterns, with one 32-bit load and no repacking.
        return tl.load(P.to(tl.pointer_type(tl.uint32))+base//2, mask0, other=0)
    return _pair(tl.load(P+base, mask0, other=0), tl.load(P+base+1, mask1, other=0))


@triton.jit
def _store_fragment(Y, r, n, a, b, c, d, M: tl.constexpr, N: tl.constexpr):
    tl.store(Y+r*N+n, a, (r < M) & (n < N))
    tl.store(Y+r*N+n+1, b, (r < M) & (n+1 < N))
    tl.store(Y+(r+8)*N+n, c, (r+8 < M) & (n < N))
    tl.store(Y+(r+8)*N+n+1, d, (r+8 < M) & (n+1 < N))


@triton.jit
def explicit_turing_mma_kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               ALIGNED: tl.constexpr):
    thread = tl.arange(0, 128)
    lane, warp = thread % 32, thread // 32
    group, pair_column = lane // 4, (lane % 4)*2
    r = tl.program_id(0)*64 + warp*16 + group
    base_n = tl.program_id(1)*32
    bn = base_n + group
    cn = base_n + pair_column
    c00=tl.full((128,),0,tl.float32); c01=tl.full((128,),0,tl.float32)
    c02=tl.full((128,),0,tl.float32); c03=tl.full((128,),0,tl.float32)
    c10=tl.full((128,),0,tl.float32); c11=tl.full((128,),0,tl.float32)
    c12=tl.full((128,),0,tl.float32); c13=tl.full((128,),0,tl.float32)
    c20=tl.full((128,),0,tl.float32); c21=tl.full((128,),0,tl.float32)
    c22=tl.full((128,),0,tl.float32); c23=tl.full((128,),0,tl.float32)
    c30=tl.full((128,),0,tl.float32); c31=tl.full((128,),0,tl.float32)
    c32=tl.full((128,),0,tl.float32); c33=tl.full((128,),0,tl.float32)
    for block in range(tl.cdiv(K,32)):
        # Preserve the K accumulation order, exposing four successive groups
        # to the compiler together for scheduling and fewer loop branches.
        for inner in tl.static_range(4):
            k = block*32 + inner*8 + pair_column
            a0 = _load_pair(X,r*K+k,(r < M) & (k < K),(r < M) & (k+1 < K),ALIGNED)
            a1 = _load_pair(X,(r+8)*K+k,(r+8 < M) & (k < K),(r+8 < M) & (k+1 < K),ALIGNED)
            b0 = _load_pair(W,bn*K+k,(bn < N) & (k < K),(bn < N) & (k+1 < K),ALIGNED)
            b1 = _load_pair(W,(bn+8)*K+k,(bn+8 < N) & (k < K),(bn+8 < N) & (k+1 < K),ALIGNED)
            b2 = _load_pair(W,(bn+16)*K+k,(bn+16 < N) & (k < K),(bn+16 < N) & (k+1 < K),ALIGNED)
            b3 = _load_pair(W,(bn+24)*K+k,(bn+24 < N) & (k < K),(bn+24 < N) & (k+1 < K),ALIGNED)
            c00,c01,c02,c03 = _mma(a0,a1,b0,c00,c01,c02,c03)
            c10,c11,c12,c13 = _mma(a0,a1,b1,c10,c11,c12,c13)
            c20,c21,c22,c23 = _mma(a0,a1,b2,c20,c21,c22,c23)
            c30,c31,c32,c33 = _mma(a0,a1,b3,c30,c31,c32,c33)
    _store_fragment(Y,r,cn,c00,c01,c02,c03,M,N)
    _store_fragment(Y,r,cn+8,c10,c11,c12,c13,M,N)
    _store_fragment(Y,r,cn+16,c20,c21,c22,c23,M,N)
    _store_fragment(Y,r,cn+24,c30,c31,c32,c33,M,N)


def explicit_linear(x, weight, *, fp32_output=False):
    if (not all(t.is_cuda and t.dtype == torch.float16 and t.ndim == 2 and t.is_contiguous()
                for t in (x,weight)) or x.shape[1] != weight.shape[1]
            or x.device != weight.device or min(*x.shape,weight.shape[0]) <= 0):
        raise ValueError('nonempty contiguous FP16 CUDA X[M,K], W[N,K] required')
    if torch.cuda.get_device_capability(x.device) != (7,5):
        raise ValueError('this isolated prototype is restricted to the T4/SM75')
    m,k = x.shape
    n = weight.shape[0]
    output = torch.empty((m,n),device=x.device,dtype=torch.float32 if fp32_output else torch.float16)
    compiled = explicit_turing_mma_kernel[(triton.cdiv(m,64),triton.cdiv(n,32))](
        x,weight,output,m,n,k,k%2==0 and x.data_ptr()%4==0 and weight.data_ptr()%4==0,
        num_warps=4,enable_fp_fusion=False)
    return output,compiled

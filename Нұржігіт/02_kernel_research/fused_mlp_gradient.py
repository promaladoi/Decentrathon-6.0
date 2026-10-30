"""Two independent tensor-core products plus their FP32 gradient sum.

Preserve the shared MLP backward's two FP16-rounded partial input gradients,
then add in FP32. This kernel is a new matmul/reduction fusion, not a renamed
SwiGLU example. It remains inactive until separate T4 correctness/performance
measurements select and admit a configuration.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M'])
def mlp_two_product_gradient(DG,DU,WG,WU,DX,M,N:tl.constexpr,K:tl.constexpr,
                             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                             GROUP:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    width=GROUP*nn
    first=(pid//width)*GROUP
    group_m=tl.minimum(nm-first,GROUP)
    pm=first+(pid%width)%group_m
    pn=(pid%width)//group_m
    m=pm*BM+tl.arange(0,BM)
    n=pn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32)
    au=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+kk
        g=tl.load(DG+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        u=tl.load(DU+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        wg=tl.load(WG+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N),0)
        wu=tl.load(WU+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N),0)
        ag=tl.dot(g,wg,ag)
        au=tl.dot(u,wu,au)
    # Both native matrix products round to FP16 before autograd sums their
    # FP32 casts. Combining the dot accumulators before rounding would differ.
    dx=ag.to(tl.float16).to(tl.float32)+au.to(tl.float16).to(tl.float32)
    tl.store(DX+m[:,None]*N+n[None,:],dx,(m[:,None]<M)&(n[None,:]<N))


def fused_gradient(dg,du,wg,wu,config=(32,64,64,4)):
    bm,bn,bk,warps=config
    m,k=dg.shape; n=wg.shape[1]
    if (not all(t.is_cuda and t.dtype==torch.float16 and t.is_contiguous()
                for t in (dg,du,wg,wu)) or du.shape!=dg.shape or
            wg.shape!=wu.shape or wg.shape[0]!=k):
        raise ValueError('contiguous FP16 CUDA matrices with matching dimensions required')
    out=torch.empty((m,n),device=dg.device,dtype=torch.float32)
    mlp_two_product_gradient[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        dg,du,wg,wu,out,m,n,k,bm,bn,bk,8,num_warps=warps)
    return out

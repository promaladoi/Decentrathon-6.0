"""Inactive research: real-row Philox mapping for native PyTorch 2.11 VEC4.

Not imported by solution.apply(). This screen uses Triton's Philox primitive,
not a copied/ready-made dropout kernel. The coordinate mapping follows the
published PyTorch native implementation; float conversion follows cuRAND.
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu
https://docs.nvidia.com/cuda/curand/curand/curand__uniform_8h_source.html

No generator mutation occurs in the kernel or gather(). The isolated screen
supplies explicit seed/counter values. A production integration would also
need a qualified reservation contract and complete original admission.
"""
import torch
import triton
import triton.language as tl
from triton.language.random import philox


@triton.jit
def real_philox_gather_kernel(X, Rows, Out, Mask, Groups,
                             Seed, Counter, THREADS: tl.constexpr,
                             WIDTH: tl.constexpr, KEEP: tl.constexpr,
                             SCALE: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid = group < Groups
    flat = group*4
    row = tl.load(Rows+flat//WIDTH,valid,other=0)
    full_group = (row*WIDTH+flat%WIDTH)//4
    thread = (full_group%THREADS).to(tl.uint32)
    counter = Counter.to(tl.uint64)+full_group.to(tl.uint64)//THREADS
    zero = tl.full((BLOCK,),0,tl.uint32)
    u0,u1,u2,u3 = philox(Seed,counter.to(tl.uint32),
                         (counter>>32).to(tl.uint32),thread,zero)
    component = tl.arange(0,4)
    integer = tl.where(component[None,:]==0,u0[:,None],
              tl.where(component[None,:]==1,u1[:,None],
              tl.where(component[None,:]==2,u2[:,None],u3[:,None])))
    # Explicit FMA matches the native cuRAND uniform conversion. Arithmetic
    # applying the mask still has its two separate FP32 rounding boundaries.
    uniform = tl.fma(integer.to(tl.float32),2.3283064365386963e-10,
                     1.1641532182693481e-10)
    kept = (uniform < KEEP) & valid[:,None]
    positions = flat[:,None]+component[None,:]
    value = tl.load(X+positions,valid[:,None],other=0).to(tl.float32)
    masked = value*kept.to(tl.float32)
    tl.store(Out+positions,masked*SCALE,valid[:,None])
    bits = tl.reshape(kept.to(tl.int32),(BLOCK//2,8))
    packed = tl.sum(bits*(1<<tl.arange(0,8)[None,:]),axis=1)
    byte = tl.program_id(0)*(BLOCK//2)+tl.arange(0,BLOCK//2)
    tl.store(Mask+byte,packed.to(tl.uint8),byte<(Groups+1)//2)


def geometry(total,device):
    props=torch.cuda.get_device_properties(device)
    assert props.major==7 and props.minor==5 and 'T4' in props.name
    # Turing SM75 supports 32 resident warps = 1024 threads per SM.
    grid=min(props.multi_processor_count*4,triton.cdiv(total,256))
    threads=grid*256
    return threads,triton.cdiv(total,threads*4)*4


def gather(x,rows,batch,sequence,p,seed,offset,output_dtype=torch.float32):
    from solution.dropout import _scales,_float32
    assert x.is_cuda and x.is_contiguous() and x.dtype==torch.float32
    assert x.ndim==2 and x.shape[1]%4==0 and x.numel()>0
    assert rows.is_contiguous() and rows.dtype==torch.int64
    assert 0<p<1 and offset%4==0
    threads,increment=geometry(batch*sequence*x.shape[1],x.device)
    out=torch.empty(x.shape,device=x.device,dtype=output_dtype)
    mask=torch.empty((triton.cdiv(x.numel(),8),),device=x.device,dtype=torch.uint8)
    groups=x.numel()//4
    real_philox_gather_kernel[(triton.cdiv(groups,256),)](
        x,rows,out,mask,groups,seed,offset//4,threads,x.shape[1],
        _float32(1-float(p)),_scales(p)[0],256,
        num_warps=4,enable_fp_fusion=False)
    return out,mask,increment

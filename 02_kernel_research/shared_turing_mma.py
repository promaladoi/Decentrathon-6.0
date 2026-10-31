"""Original v58 SM75 pipelined shared-memory MMA; not a production patch.

Four warps reuse a coalesced FP16 A/W tile in statically allocated shared
memory. Full-address XOR preserves aligned 16-byte rows for ldmatrix.
Global input vectors for the next K tile remain in registers while the current
tile executes. All buffer reuse is protected by uniform CTA barriers.
Explicit PTX m16n8k8 uses FP32 accumulators. This first dense prototype must
beat native GEMM before any NF4 fusion or training integration is attempted.
No vendor implementation or third-party kernel library is copied.

Primary instruction/fragment reference:
https://docs.nvidia.com/cuda/archive/11.0_GA/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
"""
import functools
import torch
import triton
import triton.language as tl


@functools.lru_cache(None)
def assembly(block_m, fp32_output):
    assert block_m in (64, 128)
    fragments = block_m // 64
    count = fragments * 32
    a_bytes = block_m * 32 * 2
    lines = ['{', f'.shared .align 16 .b8 smem[{a_bytes + 4096}];',
        '.reg .u32 tid, lane, warp, group, pair, bm, bn, szM, szN, szK, posK, nextK;',
        '.reg .u32 sbase, idx, row, col, sw, sw2, dst, gRow, lin, tmp, aAdr, bAdr;',
        '.reg .u32 outRow, outCol;', '.reg .b32 b0;', f'.reg .b32 af<{fragments*2}>;',
        '.reg .u64 byteOffset, address;', '.reg .pred valid, vr, vc, more;',
        f'.reg .b32 pa<{block_m//32*4}>;', '.reg .b32 pw<8>;',
        f'.reg .f32 acc<{count}>;', '.reg .b16 half;',
        'mov.u32 tid, %tid.x;', 'and.b32 lane, tid, 31;', 'shr.u32 warp, tid, 5;',
        'shr.u32 group, lane, 2;', 'and.b32 pair, lane, 3;',
        f'mul.lo.u32 bm, $7, {block_m};', 'mul.lo.u32 bn, $8, 64;',
        'mov.u32 szM, $4;', 'mov.u32 szN, $5;', 'mov.u32 szK, $6;',
        'mov.u32 sbase, smem;']
    lines.extend(f'mov.f32 acc{i}, 0f00000000;' for i in range(count))
    def load_vectors(position):
        code=[]
        for prefix, tile_rows, ptr, row_base, bound in (
                ('pa',block_m,'$1','bm','szM'),('pw',64,'$2','bn','szN')):
            for vector in range(tile_rows//32):
                regs=', '.join(prefix+str(vector*4+i) for i in range(4))
                code += [f'// prefetch {prefix} {vector}',
                f'mad.lo.u32 idx, tid, 4, {vector * 512};',
                'shr.u32 row, idx, 4;', 'and.b32 col, idx, 15;',
                f'add.u32 gRow, {row_base}, row;',
                f'mad.lo.u32 lin, gRow, szK, {position};',
                'mad.lo.u32 lin, col, 2, lin;', 'mul.wide.u32 byteOffset, lin, 2;',
                f'add.u64 address, {ptr}, byteOffset;', f'setp.lt.u32 valid, gRow, {bound};',
                *[f'mov.b32 {prefix}{vector*4+i}, 0;' for i in range(4)],
                f'@valid ld.global.v4.b32 {{{regs}}}, [address];']
        return code

    def store_vectors():
        code=[]
        for prefix,tile_rows,shared_offset in (('pa',block_m,0),('pw',64,a_bytes)):
            for vector in range(tile_rows//32):
                regs=', '.join(prefix+str(vector*4+i) for i in range(4))
                code += [f'mad.lo.u32 idx, tid, 4, {vector*512};',
                'shr.u32 row, idx, 4;', 'and.b32 tmp, row, 7;',
                'shl.b32 tmp, tmp, 2;', 'xor.b32 dst, idx, tmp;',
                'shl.b32 dst, dst, 2;',
                f'add.u32 dst, dst, {shared_offset};', 'add.u32 dst, dst, sbase;',
                f'st.shared.v4.b32 [dst], {{{regs}}};']
        return code

    lines += ['mov.u32 posK, 0;', *load_vectors('posK'), *store_vectors(),
        'bar.sync 0;', 'SHARED_K_LOOP:', 'add.u32 nextK, posK, 32;',
        'setp.lt.u32 more, nextK, szK;', '@!more bra SHARED_COMPUTE;']
    lines += load_vectors('nextK')
    lines += ['SHARED_COMPUTE:']
    for k_fragment in range(4):
        for m_fragment in range(fragments):
            lines += ['and.b32 row, lane, 15;',
                f'mad.lo.u32 row, warp, {block_m//4}, row;',
                f'add.u32 row, row, {m_fragment * 16};',
                f'mad.lo.u32 aAdr, row, 16, {k_fragment*4};',
                'and.b32 tmp, row, 7;', 'shl.b32 tmp, tmp, 2;',
                'xor.b32 aAdr, aAdr, tmp;', 'shl.b32 aAdr, aAdr, 2;',
                'add.u32 aAdr, aAdr, sbase;',
                f'ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{af{m_fragment*2}, af{m_fragment*2+1}}}, [aAdr];']
        for n_fragment in range(8):
            lines += ['and.b32 row, lane, 7;', f'add.u32 row, row, {n_fragment*8};',
                f'mad.lo.u32 bAdr, row, 16, {k_fragment*4};',
                'and.b32 tmp, row, 7;', 'shl.b32 tmp, tmp, 2;',
                'xor.b32 bAdr, bAdr, tmp;', 'shl.b32 bAdr, bAdr, 2;',
                f'add.u32 bAdr, bAdr, {a_bytes};', 'add.u32 bAdr, bAdr, sbase;',
                'ldmatrix.sync.aligned.m8n8.x1.shared.b16 {b0}, [bAdr];']
            for m_fragment in range(fragments):
                regs = ', '.join('acc'+str(m_fragment*32+n_fragment*4+i) for i in range(4))
                lines += [
                    f'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 '
                    f'{{{regs}}}, {{af{m_fragment*2}, af{m_fragment*2+1}}}, {{b0}}, {{{regs}}};']
    # A second barrier prevents the next iteration overwriting another warp's
    # still-live tile. All threads execute the same nonempty uniform K loop.
    lines += ['bar.sync 0;', '@!more bra SHARED_DONE;', *store_vectors(),
        'bar.sync 0;', 'mov.u32 posK, nextK;', 'bra SHARED_K_LOOP;', 'SHARED_DONE:']
    for m_fragment in range(fragments):
        for n_fragment in range(8):
            for i, (dr, dc) in enumerate(((0,0),(0,1),(8,0),(8,1))):
                acc = 'acc'+str(m_fragment*32+n_fragment*4+i)
                lines += [f'mad.lo.u32 outRow, warp, {block_m // 4}, group;',
                    f'add.u32 outRow, outRow, {m_fragment*16+dr};',
                    'add.u32 outRow, outRow, bm;',
                    f'mad.lo.u32 outCol, pair, 2, {n_fragment*8+dc};',
                    'add.u32 outCol, outCol, bn;',
                    'setp.lt.u32 vr, outRow, szM;', 'setp.lt.u32 vc, outCol, szN;',
                    'and.pred valid, vr, vc;', 'mad.lo.u32 lin, outRow, szN, outCol;',
                    f'mul.wide.u32 byteOffset, lin, {4 if fp32_output else 2};',
                    'add.u64 address, $3, byteOffset;']
                if fp32_output:
                    lines.append(f'@valid st.global.f32 [address], {acc};')
                else:
                    lines += [f'cvt.rn.f16.f32 half, {acc};', '@valid st.global.b16 [address], half;']
    lines += ['mov.u32 $0, 0;', '}']
    return '\n'.join(lines)


@triton.jit(do_not_specialize=['M', 'N', 'K'])
def shared_turing_mma_kernel(X, W, Y, M, N, K,
                              BM: tl.constexpr, GROUP_M: tl.constexpr,
                              ASM: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, 64)
    per_group = GROUP_M * nn
    group = pid // per_group
    first_m = group * GROUP_M
    active_m = tl.minimum(nm-first_m, GROUP_M)
    pid_m = first_m+(pid % per_group) % active_m
    pid_n = (pid % per_group) // active_m
    unused = tl.inline_asm_elementwise(ASM,
        constraints='=r,l,l,l,r,r,r,r,r,r',
        args=[X.to(tl.uint64),W.to(tl.uint64),Y.to(tl.uint64),
              M,N,K,pid_m,pid_n,tl.arange(0,128)],dtype=tl.int32,
        is_pure=False,pack=1)


def shared_linear(x, weight, *, block_m=64, group_m=8, fp32_output=False):
    if (not all(t.is_cuda and t.dtype == torch.float16 and t.ndim == 2 and t.is_contiguous()
                for t in (x,weight)) or x.shape[1] != weight.shape[1]
            or x.device != weight.device or min(*x.shape,weight.shape[0]) <= 0
            or x.shape[1] % 32 or max(x.numel(), weight.numel(),x.shape[0]*weight.shape[0]) >= 2**31
            or block_m not in (64,128) or group_m not in (1,4,8,16)):
        raise ValueError('contiguous CUDA FP16 X[M,K], W[N,K], K multiple of 32 and safe dimensions required')
    if torch.cuda.get_device_capability(x.device) != (7,5):
        raise ValueError('isolated SM75 prototype')
    m,k = x.shape
    n = weight.shape[0]
    y = torch.empty((m,n),device=x.device,dtype=torch.float32 if fp32_output else torch.float16)
    compiled = shared_turing_mma_kernel[(triton.cdiv(m,block_m)*triton.cdiv(n,64),)](
        x,weight,y,m,n,k,block_m,group_m,assembly(block_m,fp32_output),
        num_warps=4,enable_fp_fusion=False)
    return y,compiled

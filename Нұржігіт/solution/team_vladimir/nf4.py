"""NF4 dequantization in Triton for bitsandbytes' 4-bit linear layers, forward and backward.

Every Linear4bit in the model (q/k/v/o and gate/up/down in all 28 layers) unpacks its NF4 weight
to fp16 once in forward and once more in backward (for the gradient of its input). bitsandbytes
does this with its own CUDA kernels; with double quantization (Soup's 4-bit config) the per-block
scales are themselves 8-bit and take an extra dequantize launch plus an add before the weight
can be unpacked. Here one launch does all of it: per output element it reads its nibble, looks up
the NF4 code, rebuilds the block scale from the 8-bit scale, the second-level scale and the offset,
and writes the fp16 weight. The matmul itself stays in cuBLAS (F.linear), exactly as bitsandbytes'
dequantize + F.linear path, and no dense weight is kept between forward and backward.

Layout (bitsandbytes 0.50.2): weight [N, K] flattened, two values per byte, the even element in the
high nibble; scale of element i = absmax[i // blocksize]; with nested quantization
absmax[b] = code2[absmax_u8[b]] * absmax2[b // 256] + offset.
Idea: bitsandbytes' NF4 format (Dettmers et al., QLoRA, https://arxiv.org/abs/2305.14314);
code written by us.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit
def nf4_dequant_kernel(PACKED, CODE, ABSMAX, ABSMAX2, CODE2, OFFSET, OUT, n,
                       QBLOCK: tl.constexpr, NESTED: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    byte = tl.load(PACKED + idx // 2, mask=mask, other=0).to(tl.int32)
    nibble = (byte >> ((1 - (idx % 2)) * 4).to(tl.int32)) & 0xF
    value = tl.load(CODE + nibble, mask=mask, other=0.0)
    qblock = idx // QBLOCK
    if NESTED:
        q8 = tl.load(ABSMAX + qblock, mask=mask, other=0).to(tl.int32)
        scale = tl.load(CODE2 + q8, mask=mask, other=0.0) * \
            tl.load(ABSMAX2 + qblock // 256, mask=mask, other=0.0) + tl.load(OFFSET)
    else:
        scale = tl.load(ABSMAX + qblock, mask=mask, other=0.0)
    tl.store(OUT + idx, (value * scale).to(OUT.dtype.element_ty), mask=mask)


def supported(state):
    if state is None or state.quant_type != "nf4" or len(state.shape) != 2:
        return False
    if getattr(state, "code", None) is None or state.code.numel() != 16:
        return False
    if state.nested:
        s2 = state.state2
        return s2 is not None and s2.blocksize == 256 and s2.code is not None and \
            torch.is_tensor(state.offset)
    return True


def dequantize_nf4(packed, state, dtype):
    """The [N, K] weight of a Linear4bit in `dtype`, unpacked by one Triton launch."""
    out = torch.empty(tuple(state.shape), device=packed.device, dtype=dtype)
    n = out.numel()
    packed = packed.reshape(-1)
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    code = state.code.to(device=packed.device, dtype=torch.float32)
    if state.nested:
        s2 = state.state2
        args = (state.absmax, s2.absmax.float(), s2.code.float(), state.offset.float().reshape(1))
    else:
        absmax = state.absmax.float()
        args = (absmax, absmax, code, code)  # unused when NESTED is off
    nf4_dequant_kernel[(triton.cdiv(n, BLOCK),)](packed, code, *args, out, n,
                                                 QBLOCK=state.blocksize, NESTED=bool(state.nested),
                                                 BLOCK=BLOCK, num_warps=8)
    return out


class MatMulNF4(torch.autograd.Function):
    """A @ W.T + bias with W unpacked by our kernel; the same math as bitsandbytes' MatMul4Bit
    (dequantize, then F.linear forward and grad_output @ W backward; no gradient for W)."""

    @staticmethod
    def forward(ctx, A, packed, bias, state):
        weight = dequantize_nf4(packed, state, A.dtype)
        out = torch.nn.functional.linear(A, weight, bias)
        ctx.state, ctx.packed = state, packed
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_A = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(ctx.packed, ctx.state, grad_output.dtype)
            grad_A = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        return grad_A, None, grad_bias, None


def make_matmul_4bit(original):
    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if out is not None or not A.is_cuda or A.numel() == 0 or not supported(quant_state) or \
                A.dtype not in (torch.float16, torch.float32) or A.shape[-1] != quant_state.shape[1]:
            return original(A, B, quant_state, out=out, bias=bias)
        return MatMulNF4.apply(A, B, bias, quant_state)

    return matmul_4bit


def patch_bnb_matmul_4bit():
    """Route bitsandbytes.matmul_4bit (called by Linear4bit.forward, so by every 4-bit layer and
    by PEFT's LoRA layers through their base layer) through MatMulNF4."""
    import bitsandbytes

    bitsandbytes.matmul_4bit = make_matmul_4bit(bitsandbytes.matmul_4bit)


def warmup():
    """Compile the fp16, double-quantized variant every 4-bit layer launches; see warmup.py."""
    f32, u8 = torch.float32, torch.uint8
    nf4_dequant_kernel.warmup(u8, f32, u8, f32, f32, f32, torch.float16, 16 * 1536, grid=(1,),
                              QBLOCK=64, NESTED=True, BLOCK=BLOCK, num_warps=8)

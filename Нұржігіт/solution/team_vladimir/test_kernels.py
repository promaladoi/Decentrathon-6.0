"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.team_vladimir.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.team_vladimir.rmsnorm import rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    # transformers' Qwen2RMSNorm.forward, with the weight in fp32 as after k-bit preparation
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    ref = weight * h.to(dtype)
    ref.backward(grad)
    ref_dx = x.grad.float()
    x.grad = None

    out = rmsnorm(x, weight, eps)
    out.backward(grad)
    assert out.dtype == ref.dtype
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(8960, 1536), (256, 1536), (96, 200)])
def test_nf4_dequant_matches_bitsandbytes(nested, shape):
    import bitsandbytes.functional as bnbf
    from solution.team_vladimir.nf4 import dequantize_nf4

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, state = bnbf.quantize_4bit(weight, blocksize=64, compress_statistics=nested,
                                       quant_type="nf4")
    for dtype in (torch.float16, torch.float32):
        ref = bnbf.dequantize_4bit(packed, state).to(dtype)
        out = dequantize_nf4(packed, state, dtype)
        assert out.shape == ref.shape and out.dtype == dtype
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("bias", [True, False])
def test_nf4_matmul_matches_bitsandbytes(bias):
    import bitsandbytes.functional as bnbf
    from bitsandbytes.autograd._functions import matmul_4bit as reference
    from solution.team_vladimir.nf4 import MatMulNF4

    torch.manual_seed(0)
    n_out, n_in = 1024, 1536
    weight = torch.randn(n_out, n_in, device="cuda", dtype=torch.float16) * 0.02
    packed, state = bnbf.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                       quant_type="nf4")
    b = torch.randn(n_out, device="cuda", dtype=torch.float16) if bias else None  # frozen, as in Qwen
    x = torch.randn(4, 300, n_in, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(4, 300, n_out, device="cuda", dtype=torch.float16)

    ref = reference(x, packed, state, bias=b)
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    out = MatMulNF4.apply(x, packed, b, state)
    out.backward(grad)
    # bitsandbytes may take its fused MMA kernel here: same math, other accumulation order
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-2, atol=1e-2)

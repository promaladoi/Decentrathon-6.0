"""NF4 register decode boundaries and opt-in GEMM numerical qualification."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def _synthetic(shape=(17, 33), nested=True, state_dtype=torch.float16, block=64):
    n, k = shape
    elements = n * k
    packed = (torch.arange((elements + 1) // 2, device="cuda") % 256).to(torch.uint8)
    code = (torch.arange(16, device="cuda", dtype=torch.float32) - 8) / 8
    count = (elements + block - 1) // block
    state = SimpleNamespace(shape=torch.Size(shape), dtype=state_dtype, blocksize=block,
                            code=code, quant_type="nf4", nested=nested)
    if nested:
        state.absmax = (torch.arange(count, device="cuda") * 17 % 256).to(torch.uint8)
        state.state2 = SimpleNamespace(
            blocksize=256, nested=False, dtype=torch.float32,
            code=torch.linspace(-.75, .75, 256, device="cuda"),
            absmax=torch.full(((count + 255) // 256,), .7, device="cuda"))
        state.offset = torch.tensor([.8125], device="cuda")
    else:
        state.absmax = (torch.arange(count, device="cuda", dtype=torch.float32) % 5 + 1) / 8
    return packed, state


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("shape", [(17, 33), (16, 64)])
def test_identity_input_recovers_decoded_weight_with_exact_nibble_and_tail_mapping(
        shape, nested, state_dtype, transpose):
    from solution.nf4 import _decode, dequantize_nf4
    from solution.nf4_gemm import fused_nf4_gemm

    packed, state = _synthetic(shape, nested, state_dtype)
    reduction = shape[1 if transpose else 0]
    a = torch.eye(reduction, device="cuda", dtype=torch.float16)
    expected = (_decode(packed, state, torch.float16).t() if transpose
                else dequantize_nf4(packed, state, torch.float16))
    actual = fused_nf4_gemm(a, packed, state, transpose_weight=transpose)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nested_scale_multiply_and_offset_add_are_not_contracted():
    from solution.nf4_gemm import fused_nf4_gemm

    packed, state = _synthetic((17, 33), True)
    state.state2.code.fill_(.876543)
    state.state2.absmax.fill_(12345.678)
    # Separate FP32 product then its negative gives exactly zero. Fusing that
    # multiply/add can instead expose a nonzero product-rounding residual.
    state.offset.copy_(-(state.state2.code[:1] * state.state2.absmax[:1]))
    a = torch.eye(33, device="cuda", dtype=torch.float16)
    actual = fused_nf4_gemm(a, packed, state)
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("shape,m", [((8960, 1536), 17), ((1536, 8960), 33)])
def test_real_mlp_shapes_match_existing_decoder_plus_cublas(transpose, shape, m):
    import bitsandbytes.functional as bnbf
    from solution.nf4_gemm import fused_nf4_gemm, reference_nf4_gemm

    torch.manual_seed(771)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16) * .02
    packed, state = bnbf.quantize_4bit(weight, blocksize=64,
                                       compress_statistics=True, quant_type="nf4")
    del weight
    reduction = shape[1 if transpose else 0]
    a = (torch.randn((m, reduction * 2), device="cuda", dtype=torch.float16) * .2)[:, ::2]
    expected = reference_nf4_gemm(a, packed, state, transpose_weight=transpose)
    state_before = torch.cuda.get_rng_state()
    actual = fused_nf4_gemm(a, packed, state, transpose_weight=transpose)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert float(relative) < 2e-3
    assert torch.equal(state_before, torch.cuda.get_rng_state())


def test_opt_in_linear_retained_and_higher_order_gradients_match_torch():
    from solution.nf4_gemm import fused_nf4_linear, reference_nf4_gemm

    packed, state = _synthetic((64, 96), True)
    x = (torch.randn((7, 96), device="cuda", dtype=torch.float16) * .1).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    actual = fused_nf4_linear(x, packed, state)
    expected = reference_nf4_gemm(reference, packed, state)
    upstream = torch.randn_like(actual) * .1
    first, = torch.autograd.grad(actual, x, upstream, retain_graph=True)
    second, = torch.autograd.grad(actual, x, upstream)
    expected_dx, = torch.autograd.grad(expected, reference, upstream)
    torch.testing.assert_close(first, expected_dx, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    probe = torch.randn((7, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    result = fused_nf4_linear(x, packed, state)
    dx, = torch.autograd.grad(result, x, probe, create_graph=True)
    mixed, = torch.autograd.grad(dx.float().sum(), probe)
    native_dx = reference_nf4_gemm(probe, packed, state, transpose_weight=False)
    expected_mixed, = torch.autograd.grad(native_dx.float().sum(), probe)
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)


def test_saved_quantization_tensor_mutation_keeps_version_checking():
    from solution.nf4_gemm import fused_nf4_linear

    packed, state = _synthetic((32, 64), True)
    x = torch.randn((3, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    y = fused_nf4_linear(x, packed, state)
    state.state2.absmax.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.float().sum().backward()

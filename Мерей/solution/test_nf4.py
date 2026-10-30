"""NF4 decoding and the real bitsandbytes input-gradient integration."""

import inspect

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def _modules():
    pytest.importorskip("triton")
    functional = pytest.importorskip("bitsandbytes.functional")
    from solution import nf4

    return functional, nf4, inspect.unwrap(functional.dequantize_4bit)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(17, 65), (128, 1536), (1536, 1536)])
def test_nf4_decode_matches_bitsandbytes(dtype, nested, shape):
    functional, nf4, original = _modules()
    torch.manual_seed(121)
    weights = torch.randn(shape, device="cuda", dtype=dtype)
    packed, state = functional.quantize_4bit(weights, quant_type="nf4", compress_statistics=nested)
    before = packed.clone()
    expected = original(packed, state)
    actual = nf4.dequantize_nf4(packed, state)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(packed, before, rtol=0, atol=0)
    # Preserve the stock compatibility convention for transposed packed input.
    torch.testing.assert_close(nf4.dequantize_nf4(packed.t(), state),
                               original(packed.t(), state), rtol=0, atol=0)


@pytest.mark.parametrize("blocksize", [32, 64, 128, 256])
def test_nf4_nibble_order_and_quant_block_boundaries(blocksize):
    functional, nf4, original = _modules()
    packed = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8).view(-1, 1)
    code = functional.get_4bit_type("nf4", device="cuda")
    scale = torch.arange(1, 512 // blocksize + 1, device="cuda", dtype=torch.float32)
    state = functional.QuantState(absmax=scale, code=code, shape=(8, 64),
                                  dtype=torch.float32, blocksize=blocksize, quant_type="nf4")
    indices = torch.stack((packed.flatten() >> 4, packed.flatten() & 15), dim=-1).flatten().long()
    expected = (code[indices] * scale.repeat_interleave(blocksize)).view(8, 64)
    actual = nf4.dequantize_nf4(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, original(packed, state), rtol=0, atol=0)


def test_nf4_patch_is_idempotent_and_keeps_unsupported_fallbacks(monkeypatch):
    functional, nf4, _ = _modules()
    calls = []
    marker = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(functional, "dequantize_4bit", original)
    nf4.patch_nf4_dequant()
    first = functional.dequantize_4bit
    nf4.patch_nf4_dequant()
    assert functional.dequantize_4bit is first
    assert functional.dequantize_4bit(torch.zeros((4, 1), dtype=torch.uint8)) is marker
    assert functional.dequantize_4bit(object(), quant_state=object()) is marker
    packed, state = functional.quantize_4bit(torch.randn((8, 64), device="cuda"), quant_type="nf4")
    out = torch.empty((8, 64), device="cuda")
    assert functional.dequantize_4bit(packed, state, out=out) is marker
    assert len(calls) == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nf4_kernel_really_runs_in_matmul_backward(monkeypatch, dtype):
    functional, nf4, original = _modules()
    bnb = pytest.importorskip("bitsandbytes")
    monkeypatch.setattr(functional, "dequantize_4bit", original)
    torch.manual_seed(122)
    weights = torch.randn((48, 64), device="cuda", dtype=dtype)
    packed, state = functional.quantize_4bit(weights, quant_type="nf4", compress_statistics=True)
    reference = torch.randn((7, 64), device="cuda", dtype=dtype, requires_grad=True)
    actual_input = reference.detach().clone().requires_grad_(True)
    upstream = torch.randn((7, 48), device="cuda", dtype=dtype)
    expected = bnb.matmul_4bit(reference, packed.t(), state)
    expected.backward(upstream)
    launches = []
    kernel_run = nf4.soup_nf4_unpack_kernel.run

    def count(*args, **kwargs):
        launches.append(1)
        return kernel_run(*args, **kwargs)

    monkeypatch.setattr(nf4.soup_nf4_unpack_kernel, "run", count)
    nf4.patch_nf4_dequant()
    output = bnb.matmul_4bit(actual_input, packed.t(), state)
    output.backward(upstream)
    assert launches, "NF4 kernel must supply the input-gradient matrix product"
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, reference.grad, rtol=0, atol=0)

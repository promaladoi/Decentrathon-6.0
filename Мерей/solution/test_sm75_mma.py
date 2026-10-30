"""Physical-lane fragment qualification for the isolated SM75 MMA proof."""

import re

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM75 CUDA")


def _require_sm75():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("proof specifically qualifies SM75")


def test_dense_integer_fragments_and_nonzero_accumulator_are_exact():
    _require_sm75()
    from solution.sm75_mma import mma_16x8x8

    a = ((torch.arange(128, device="cuda") % 5) - 2).half().reshape(16, 8)
    b = ((torch.arange(64, device="cuda") % 7) - 3).half().reshape(8, 8)
    c = torch.arange(128, device="cuda", dtype=torch.float32).reshape(16, 8) / 8
    before = [value.clone() for value in (a, b, c)]
    result = mma_16x8x8(a, b, c)
    torch.testing.assert_close(result, a.float() @ b.float() + c, rtol=0, atol=0)
    for actual, original in zip((a, b, c), before):
        assert torch.equal(actual, original)


@pytest.mark.parametrize("row,k", [(0, 0), (7, 7), (8, 0), (15, 7), (9, 3)])
def test_one_hot_a_selects_exact_b_row_in_all_fragment_quadrants(row, k):
    _require_sm75()
    from solution.sm75_mma import mma_16x8x8

    a = torch.zeros((16, 8), device="cuda", dtype=torch.float16)
    a[row, k] = 1
    b = torch.arange(64, device="cuda", dtype=torch.float16).reshape(8, 8) - 23
    result = mma_16x8x8(a, b)
    expected = torch.zeros((16, 8), device="cuda")
    expected[row] = b[k].float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("column,k", [(0, 0), (7, 7), (5, 3)])
def test_one_hot_b_selects_exact_a_column(column, k):
    _require_sm75()
    from solution.sm75_mma import mma_16x8x8

    a = torch.arange(128, device="cuda", dtype=torch.float16).reshape(16, 8) - 37
    b = torch.zeros((8, 8), device="cuda", dtype=torch.float16)
    b[k, column] = 1
    result = mma_16x8x8(a, b)
    expected = torch.zeros((16, 8), device="cuda")
    expected[:, column] = a[:, k].float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_random_fp16_products_and_fp32_accumulation_match_native_reference():
    _require_sm75()
    from solution.sm75_mma import mma_16x8x8

    torch.manual_seed(778)
    a = torch.randn((16, 8), device="cuda", dtype=torch.float16) * .2
    b = torch.randn((8, 8), device="cuda", dtype=torch.float16) * .2
    c = torch.randn((16, 8), device="cuda", dtype=torch.float32) * .1
    state = torch.cuda.get_rng_state()
    actual = mma_16x8x8(a, b, c)
    expected = a.float() @ b.float() + c
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_offset_views_do_not_write_past_output_or_mutate_input_guards():
    _require_sm75()
    from solution.sm75_mma import mma_16x8x8

    a_storage = torch.full((130,), -19, device="cuda", dtype=torch.float16)
    b_storage = torch.full((66,), -23, device="cuda", dtype=torch.float16)
    d_storage = torch.full((130,), -29, device="cuda", dtype=torch.float32)
    a, b, out = a_storage[1:-1].reshape(16, 8), b_storage[1:-1].reshape(8, 8), d_storage[1:-1].reshape(16, 8)
    a.copy_(torch.arange(128, device="cuda", dtype=torch.float16).reshape(16, 8) % 7)
    b.copy_(torch.arange(64, device="cuda", dtype=torch.float16).reshape(8, 8) % 5)
    result = mma_16x8x8(a, b, out=out)
    assert result is out
    torch.testing.assert_close(result, a.float() @ b.float(), rtol=0, atol=0)
    assert torch.equal(a_storage[[0, -1]], torch.tensor([-19, -19], device="cuda", dtype=torch.float16))
    assert torch.equal(b_storage[[0, -1]], torch.tensor([-23, -23], device="cuda", dtype=torch.float16))
    assert torch.equal(d_storage[[0, -1]], torch.tensor([-29, -29], device="cuda", dtype=torch.float32))


def test_generated_ptx_contains_real_mma_instruction_and_one_warp_launch():
    _require_sm75()
    from solution.sm75_mma import sm75_mma_kernel

    a = torch.zeros((16, 8), device="cuda", dtype=torch.float16)
    b = torch.zeros((8, 8), device="cuda", dtype=torch.float16)
    c = torch.zeros((16, 8), device="cuda")
    d = torch.empty_like(c)
    compiled = sm75_mma_kernel[(1,)](a, b, c, d, num_warps=1)
    ptx = compiled.asm["ptx"]
    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in ptx
    assert re.search(r"\.(?:reqntid|maxntid)\s+32(?:\s*,|\s|$)", ptx)
    torch.testing.assert_close(d, c, rtol=0, atol=0)

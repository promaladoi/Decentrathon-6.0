"""Focused explicit-MMA/NF4 layout, decode, precision and graph qualification."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires the task SM75 CUDA GPU')


def _weight(rows, columns, nested=True):
    import bitsandbytes.functional as bnb

    dense = torch.randn((rows, columns), device='cuda', dtype=torch.float16) * .02
    return bnb.quantize_4bit(dense, blocksize=64, compress_statistics=nested, quant_type='nf4')


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('shape', [(71, 64), (257, 128)])
@pytest.mark.parametrize('tile_name', ['m32n64k32s1', 'm32n128k32s1'])
def test_basis_decode_and_masks_are_exact_with_turing_mma_and_zero_spills(transpose, shape, tile_name):
    from solution.nf4 import dequantize_nf4
    from solution.sm75_nf4_gemm import TILES, sm75_nf4_gemm

    torch.manual_seed(986)
    packed, state = _weight(*shape)
    reduction = shape[1 if transpose else 0]
    selected = torch.tensor([0, 1, min(63, reduction - 1), min(64, reduction - 1), reduction - 1],
                            device='cuda', dtype=torch.int64)
    a = torch.zeros((selected.numel(), reduction), device='cuda', dtype=torch.float16)
    a[torch.arange(selected.numel(), device='cuda'), selected] = 1.0
    with torch.no_grad():
        dense = dequantize_nf4(packed, state, torch.float16)
        expected = dense.t().index_select(0, selected) if transpose else dense.index_select(0, selected)
        before = torch.cuda.get_rng_state()
        actual, compiled = sm75_nf4_gemm(a, packed, state, transpose_weight=transpose,
                                         tile=TILES[tile_name], return_compiled=True)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(before, torch.cuda.get_rng_state())
    assert 'mma.sync.aligned.m16n8k8' in compiled.asm['ptx']
    assert 'mma.sync.aligned.m16n8k16' not in compiled.asm['ptx']
    assert compiled.n_spills == 0


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_random_products_match_current_on_demand_reference(transpose, strided):
    from solution.benchmark_sm75_nf4_gemm import accuracy
    from solution.nf4_gemm import reference_nf4_gemm
    from solution.sm75_nf4_gemm import sm75_nf4_gemm

    torch.manual_seed(987)
    packed, state = _weight(71, 128)
    k = state.shape[1 if transpose else 0]
    source = torch.randn((17, k * (2 if strided else 1)), device='cuda', dtype=torch.float16) * .2
    a = source[:, ::2] if strided else source
    before = [value.clone() for value in (a, packed, state.absmax, state.code, state.state2.absmax)]
    with torch.no_grad():
        expected = reference_nf4_gemm(a, packed, state, transpose_weight=transpose)
        actual = sm75_nf4_gemm(a, packed, state, transpose_weight=transpose)
        report = accuracy(actual, expected)
    assert report['qualified'], report
    for value, original in zip((a, packed, state.absmax, state.code, state.state2.absmax), before):
        assert torch.equal(value, original)


@pytest.mark.parametrize('transpose', [False, True])
def test_higher_order_requests_preserve_the_existing_torch_graph(monkeypatch, transpose):
    import solution.sm75_nf4_gemm as fused
    from solution.nf4_gemm import reference_nf4_gemm

    class ForbiddenKernel:
        def __getitem__(self, grid):
            raise AssertionError('higher-order request must use the original Torch expression')

    monkeypatch.setattr(fused, 'sm75_nf4_gemm_kernel', ForbiddenKernel())
    torch.manual_seed(988)
    packed, state = _weight(71, 64)
    k = state.shape[1 if transpose else 0]
    a = torch.randn((3, k), device='cuda', dtype=torch.float16, requires_grad=True)
    reference_input = a.detach().clone().requires_grad_()
    with torch.enable_grad():
        actual = fused.sm75_nf4_gemm(a, packed, state, transpose_weight=transpose)
        expected = reference_nf4_gemm(reference_input, packed, state, transpose_weight=transpose)
        first, = torch.autograd.grad(actual.float().square().sum(), a, create_graph=True)
        reference_first, = torch.autograd.grad(expected.float().square().sum(), reference_input,
                                              create_graph=True)
        second, = torch.autograd.grad(first.float().sum(), a)
        reference_second, = torch.autograd.grad(reference_first.float().sum(), reference_input)
    for value, original in [(actual, expected), (first, reference_first), (second, reference_second)]:
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_private_format_guards_and_empty_rows():
    from solution.sm75_nf4_gemm import Sm75Tile, sm75_nf4_gemm

    packed, state = _weight(71, 64)
    empty = torch.empty((0, 64), device='cuda', dtype=torch.float16)
    with torch.no_grad():
        output, compiled = sm75_nf4_gemm(empty, packed, state, return_compiled=True)
    assert output.shape == (0, 71) and compiled is None
    a = torch.zeros((3, 64), device='cuda', dtype=torch.float16)
    with pytest.raises(ValueError, match='unsupported SM75'):
        sm75_nf4_gemm(a, packed, state, tile=Sm75Tile(k=64))
    plain, plain_state = _weight(71, 64, nested=False)
    with pytest.raises(ValueError, match='SM75 Gluon requires'):
        sm75_nf4_gemm(a, plain, plain_state)

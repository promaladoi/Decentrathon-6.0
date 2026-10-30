"""Qualification of the inactive direct-NF4 paired MLP gradient experiment."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _weight(k, n, *, ones=False):
    import bitsandbytes.functional as bnb

    dense = (torch.ones((k, n), device='cuda', dtype=torch.float16) if ones else
             torch.randn((k, n), device='cuda', dtype=torch.float16) * .02)
    packed, state = bnb.quantize_4bit(dense, blocksize=64, quant_type='nf4', compress_statistics=True)
    packed.quant_state = state
    return packed


@pytest.mark.parametrize('config', ['m16n32k32w4s2', 'm32n64k32w4s2'])
@pytest.mark.parametrize('m,k,n', [(1, 71, 67), (17, 257, 67), (33, 64, 65)])
@pytest.mark.parametrize('output_dtype', [torch.float16, torch.float32])
def test_pair_matches_native_partial_ulp_gate_with_m_kn_tails_and_nested_boundary(
        config, m, k, n, output_dtype):
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.mlp_grad_pair import mlp_grad_pair

    torch.manual_seed(955)
    gate, up = _weight(k, n), _weight(k, n)
    dg = torch.randn((m, k), device='cuda', dtype=torch.float16)
    du = torch.randn_like(dg)
    copies = [value.clone() for value in (dg, du, gate, up)]
    state = torch.cuda.get_rng_state()
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        # Independent ordinary Torch cast/add, not the experimental kernel.
        expected = (gate_part.float() + up_part.float()).to(output_dtype)
        actual = mlp_grad_pair(dg, du, gate, up, output_dtype, config)
        metrics = accuracy_metrics(actual, expected, gate_part, up_part)
    assert actual.shape == (m, n) and actual.dtype == output_dtype
    assert metrics['qualified'], metrics
    assert torch.equal(torch.cuda.get_rng_state(), state)
    for value, original in zip((dg, du, gate, up), copies):
        assert torch.equal(value, original)


@pytest.mark.parametrize('output_dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('config', ['m32n32k32w4s2', 'm16n64k128w4s1_row_scale'])
def test_each_product_is_half_rounded_before_cancellation(output_dtype, config):
    from solution.mlp_grad_pair import mlp_grad_pair

    gate, up = _weight(64, 64, ones=True), _weight(64, 64, ones=True)
    dg = torch.zeros((3, 64), device='cuda', dtype=torch.float16)
    du = torch.zeros_like(dg)
    dg[:, 0], dg[:, 1] = 1.0, 2 ** -12
    du[:, 0] = -1.0
    # The gate product is 1+2^-12, which must become half 1 before adding -1.
    # Adding the FP32 accumulators first would incorrectly retain 2^-12.
    with torch.no_grad():
        actual = mlp_grad_pair(dg, du, gate, up, output_dtype, config)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_real_task_dimensions_match_native_accuracy_gate():
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.mlp_grad_pair import mlp_grad_pair

    torch.manual_seed(956)
    gate, up = _weight(8960, 1536), _weight(8960, 1536)
    dg = torch.randn((33, 8960), device='cuda', dtype=torch.float16)
    du = torch.randn_like(dg)
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        expected = gate_part.float() + up_part.float()
        actual = mlp_grad_pair(dg, du, gate, up)
        metrics = accuracy_metrics(actual, expected, gate_part, up_part)
    assert metrics['qualified'], metrics


def test_direct_kernel_output_sentinels_survive_partial_tiles():
    import triton
    from solution.mlp_grad_pair import CONFIGS, DEFAULT_CONFIG, mlp_grad_pair_kernel

    m, k, n = 17, 71, 67
    gate, up = _weight(k, n), _weight(k, n)
    gs, us = gate.quant_state, up.quant_state
    dg = torch.randn((m, k), device='cuda', dtype=torch.float16)
    du = torch.randn_like(dg)
    storage = torch.full((m * n + 2,), 973.0, device='cuda', dtype=torch.float32)
    bm, bn, bk, warps, stages = CONFIGS[DEFAULT_CONFIG]
    mlp_grad_pair_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dg, du, gate, gs.absmax, gs.code, gs.state2.absmax, gs.state2.code, gs.offset,
        up, us.absmax, us.code, us.state2.absmax, us.state2.code, us.offset,
        storage[1:-1], m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    assert torch.equal(storage[[0, -1]], torch.tensor([973.0, 973.0], device='cuda'))


def test_empty_rows_and_unsupported_inputs_do_not_silently_drop_autograd():
    from solution.mlp_grad_pair import mlp_grad_pair

    gate, up = _weight(64, 64), _weight(64, 64)
    empty = torch.empty((0, 64), device='cuda', dtype=torch.float16)
    with torch.no_grad():
        assert mlp_grad_pair(empty, empty, gate, up).shape == (0, 64)
    dg = torch.randn((3, 64), device='cuda', dtype=torch.float16, requires_grad=True)
    du = torch.randn_like(dg)
    with torch.enable_grad(), pytest.raises(ValueError, match='MLP pair requires'):
        mlp_grad_pair(dg, du, gate, up)
    with torch.no_grad(), pytest.raises(ValueError, match='MLP pair requires'):
        mlp_grad_pair(dg.float(), du.float(), gate, up)
    with pytest.raises(ValueError, match='unknown experimental'):
        mlp_grad_pair(dg.detach(), du, gate, up, config='not-a-tile')


@pytest.mark.parametrize('k,n', [(257, 64), (71, 128)])
def test_row_scale_revision_decodes_basis_products_exactly(k, n):
    from solution.mlp_grad_pair import ROW_SCALE_CONFIG, mlp_grad_pair
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(957)
    gate, up = _weight(k, n), _weight(k, n)
    selected = torch.tensor([0, 1, min(63, k - 1), min(64, k - 1), k - 1],
                            device='cuda', dtype=torch.int64)
    dg = torch.zeros((selected.numel(), k), device='cuda', dtype=torch.float16)
    dg[torch.arange(selected.numel(), device='cuda'), selected] = 1.0
    du = torch.zeros_like(dg)
    with torch.no_grad():
        expected = dequantize_nf4(gate, gate.quant_state, torch.float16).index_select(0, selected).float()
        actual = mlp_grad_pair(dg, du, gate, up, config=ROW_SCALE_CONFIG)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_row_scale_revision_rejects_unaligned_output_width():
    from solution.mlp_grad_pair import ROW_SCALE_CONFIG, mlp_grad_pair

    gate, up = _weight(71, 67), _weight(71, 67)
    dg = torch.zeros((3, 71), device='cuda', dtype=torch.float16)
    with torch.no_grad(), pytest.raises(ValueError, match='divisible by NF4 block 64'):
        mlp_grad_pair(dg, dg, gate, up, config=ROW_SCALE_CONFIG)

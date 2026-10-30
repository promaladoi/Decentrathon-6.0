"""Strict native partial-ULP qualification for inactive SM75 paired NF4 GEMM."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM75/Triton")


def _require_sm75():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("explicit Turing layout qualification")


@pytest.mark.parametrize("config", ["m16n64k32w4s1", "m32n64k64w4s1", "m64n64k32w4s1"])
@pytest.mark.parametrize("m,k,n", [(1, 71, 64), (17, 257, 64), (33, 64, 128)])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
def test_pair_preserves_existing_strict_native_error_gate(config, m, k, n, output_dtype):
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.sm75_mlp_pair import sm75_mlp_grad_pair

    torch.manual_seed(955)
    gate, up = _weight(k, n), _weight(k, n)
    dg = torch.randn((m, k), device="cuda", dtype=torch.float16)
    du = torch.randn_like(dg)
    state = torch.cuda.get_rng_state()
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        expected = (gate_part.float() + up_part.float()).to(output_dtype)
        actual, compiled = sm75_mlp_grad_pair(dg, du, gate, up, output_dtype, config,
                                             return_compiled=True)
        metrics = accuracy_metrics(actual, expected, gate_part, up_part)
    assert metrics["qualified"], metrics
    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in compiled.asm["ptx"]
    assert torch.equal(state, torch.cuda.get_rng_state())


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
def test_separate_half_rounding_precedes_cancellation(output_dtype):
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.sm75_mlp_pair import sm75_mlp_grad_pair

    gate, up = _weight(64, 64, ones=True), _weight(64, 64, ones=True)
    dg = torch.zeros((3, 64), device="cuda", dtype=torch.float16)
    du = torch.zeros_like(dg)
    dg[:, 0], dg[:, 1], du[:, 0] = 1.0, 2 ** -12, -1.0
    with torch.no_grad():
        actual = sm75_mlp_grad_pair(dg, du, gate, up, output_dtype)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_real_mlp_k8960_n1536_keeps_native_partial_ulp_gate():
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.sm75_mlp_pair import sm75_mlp_grad_pair

    torch.manual_seed(956)
    gate, up = _weight(8960, 1536), _weight(8960, 1536)
    dg = torch.randn((33, 8960), device="cuda", dtype=torch.float16)
    du = torch.randn_like(dg)
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        actual = sm75_mlp_grad_pair(dg, du, gate, up)
        metrics = accuracy_metrics(actual, gate_part.float() + up_part.float(), gate_part, up_part)
    assert metrics["qualified"], metrics


def test_output_tail_guards_and_input_values_are_preserved():
    _require_sm75()
    import triton
    from solution.test_mlp_grad_pair import _weight
    from solution.sm75_mlp_pair import sm75_mlp_pair_kernel

    m, k, n = 17, 71, 64
    gate, up = _weight(k, n), _weight(k, n)
    gs, us = gate.quant_state, up.quant_state
    dg = torch.randn((m, k), device="cuda", dtype=torch.float16)
    du = torch.randn_like(dg)
    copies = [value.clone() for value in (dg, du, gate, up)]
    storage = torch.full((m * n + 2,), 973.0, device="cuda")
    sm75_mlp_pair_kernel[(triton.cdiv(m, 32), 1)](
        dg, du, gate, gs.absmax, gs.code, gs.state2.absmax, gs.state2.code, gs.offset,
        up, us.absmax, us.code, us.state2.absmax, us.state2.code, us.offset,
        storage[1:-1], m, n, k, BLOCK_M=32, BLOCK_K=32,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    assert torch.equal(storage[[0, -1]], torch.tensor([973.0, 973.0], device="cuda"))
    for actual, expected in zip((dg, du, gate, up), copies):
        assert torch.equal(actual, expected)


def test_empty_rows_and_unsupported_inputs_are_explicit():
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.sm75_mlp_pair import sm75_mlp_grad_pair

    gate, up = _weight(64, 64), _weight(64, 64)
    empty = torch.empty((0, 64), device="cuda", dtype=torch.float16)
    with torch.no_grad():
        assert sm75_mlp_grad_pair(empty, empty, gate, up).shape == (0, 64)
    dg = torch.randn((3, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    with pytest.raises(ValueError, match="requires dense half"):
        sm75_mlp_grad_pair(dg, dg, gate, up)
    with pytest.raises(ValueError, match="unknown SM75"):
        sm75_mlp_grad_pair(dg.detach(), dg.detach(), gate, up, config="unknown")

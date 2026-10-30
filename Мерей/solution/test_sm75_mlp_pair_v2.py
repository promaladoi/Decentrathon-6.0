"""V2 scale bits, half boundaries and unchanged native partial-ULP gate."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM75/Triton")


def _require_sm75():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("explicit Turing layout qualification")


def _scale_reference(state):
    count = (state.shape[0] * state.shape[1] + 63) // 64
    ids = torch.arange(count, device=state.code.device)
    decoded = state.state2.code[state.absmax[:count].long()] * state.state2.absmax[ids // 256]
    return decoded + state.offset


@pytest.mark.parametrize("k,n", [(71, 64), (257, 128), (8960, 1536)])
def test_predecoded_scale_vectors_are_bitwise_native_and_ephemeral(k, n):
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.nf4_scale_pair import predecode_nested_scales_pair

    torch.manual_seed(955)
    gate, up = _weight(k, n), _weight(k, n)
    state = torch.cuda.get_rng_state()
    first = predecode_nested_scales_pair(gate.quant_state, up.quant_state)
    second = predecode_nested_scales_pair(gate.quant_state, up.quant_state)
    assert first.data_ptr() != second.data_ptr()
    expected = torch.stack((_scale_reference(gate.quant_state), _scale_reference(up.quant_state)))
    assert torch.equal(first.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(second.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(state, torch.cuda.get_rng_state())
    assert first.numel() * first.element_size() == 2 * ((k * n + 63) // 64) * 4


@pytest.mark.parametrize("config", ["m32n64k32w4s1_scales", "m128n32k32w4s1_scales"])
@pytest.mark.parametrize("m,k,n", [(1, 71, 64), (17, 257, 128), (33, 64, 128)])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
def test_v2_pair_keeps_existing_strict_native_error_gate(config, m, k, n, output_dtype):
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.sm75_mlp_pair_v2 import sm75_mlp_grad_pair_v2

    torch.manual_seed(955)
    gate, up = _weight(k, n), _weight(k, n)
    dg = torch.randn((m, k), device="cuda", dtype=torch.float16)
    du = torch.randn_like(dg)
    state = torch.cuda.get_rng_state()
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        expected = (gate_part.float() + up_part.float()).to(output_dtype)
        actual, compiled = sm75_mlp_grad_pair_v2(dg, du, gate, up, output_dtype, config,
                                                return_compiled=True)
        metrics = accuracy_metrics(actual, expected, gate_part, up_part)
    assert metrics["qualified"], metrics
    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in compiled.asm["ptx"]
    assert torch.equal(state, torch.cuda.get_rng_state())


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
def test_v2_separate_half_rounding_still_precedes_cancellation(output_dtype):
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.sm75_mlp_pair_v2 import sm75_mlp_grad_pair_v2

    gate, up = _weight(64, 64, ones=True), _weight(64, 64, ones=True)
    dg = torch.zeros((3, 64), device="cuda", dtype=torch.float16)
    du = torch.zeros_like(dg)
    dg[:, 0], dg[:, 1], du[:, 0] = 1.0, 2 ** -12, -1.0
    with torch.no_grad():
        actual = sm75_mlp_grad_pair_v2(dg, du, gate, up, output_dtype)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_v2_real_mlp_dimensions_keep_native_partial_ulp_gate():
    _require_sm75()
    from solution.test_mlp_grad_pair import _weight
    from solution.benchmark_mlp_grad_pair import accuracy_metrics, reference_parts
    from solution.sm75_mlp_pair_v2 import sm75_mlp_grad_pair_v2

    torch.manual_seed(956)
    gate, up = _weight(8960, 1536), _weight(8960, 1536)
    dg = torch.randn((33, 8960), device="cuda", dtype=torch.float16)
    du = torch.randn_like(dg)
    with torch.no_grad():
        gate_part, up_part = reference_parts(dg, du, gate, up)
        actual = sm75_mlp_grad_pair_v2(dg, du, gate, up)
        metrics = accuracy_metrics(actual, gate_part.float() + up_part.float(), gate_part, up_part)
    assert metrics["qualified"], metrics

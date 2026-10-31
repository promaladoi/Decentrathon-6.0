"""Packed LoRA scaling preserves the eager half boundaries and native RNG."""

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


def _same_bits(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual[finite].view(integer), expected[finite].view(integer))


def _inputs(exceptional=False):
    batch, sequence, hq, hk, dim = 2, 5, 4, 2, 16
    rows = torch.tensor([0, 1, 2, 3, 4, 5, 6], device="cuda", dtype=torch.int64)
    tokens = rows.numel()
    leaves = [torch.randn((tokens, heads * dim), device="cuda", dtype=torch.float16)
              for heads in (hq, hq, hk, hk, hk)]
    if exceptional:
        values = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"),
                               40000.0, -40000.0, 65504.0, -65504.0, 0.000000059604645],
                              device="cuda", dtype=torch.float16)
        for value in leaves:
            value.copy_(values.repeat((value.numel() + values.numel() - 1) // values.numel())
                        [:value.numel()].reshape_as(value))
    leaves = tuple(value.requires_grad_() for value in leaves)
    positions = (rows % sequence).float()[:, None]
    inv = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    angles = torch.cat((positions * inv, positions * inv), dim=1)
    return leaves, (angles.cos(), angles.sin(), rows, batch, sequence, hq, hk, dim, hq // hk)


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.parametrize("upstream_scale", [1.0, 1024.0, 65536.0])
def test_fused_scaling_matches_separate_half_multiply_forward_backward_and_retained_graph(
        exceptional, upstream_scale):
    from solution.unpad import RoPEPacked, RoPEPackedScaled

    torch.manual_seed(842)
    actual_inputs, metadata = _inputs(exceptional)
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in actual_inputs)
    q, lq, k, v, lv = actual_inputs
    rq, rlq, rk, rv, rlv = reference_inputs
    reference = RoPEPacked.apply(rq, rlq * 2.0, rk, None, rv, rlv * 2.0, *metadata)
    before = torch.cuda.get_rng_state()
    actual = RoPEPackedScaled.apply(q, lq, k, None, v, lv, *metadata, 2.0, 2.0)
    assert torch.equal(torch.cuda.get_rng_state(), before)
    for value, expected in zip(actual, reference):
        _same_bits(value, expected)
    upstream = tuple((torch.randn_like(value) * upstream_scale) for value in actual)
    before_backward = torch.cuda.get_rng_state()
    expected_gradients = torch.autograd.grad(reference, reference_inputs, upstream)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, upstream, retain_graph=True)
    repeated = torch.autograd.grad(actual, actual_inputs, upstream)
    assert torch.equal(torch.cuda.get_rng_state(), before_backward)
    for value, expected, again in zip(actual_gradients, expected_gradients, repeated):
        assert value.dtype == expected.dtype == torch.float16
        _same_bits(value, expected)
        _same_bits(again, value)


@pytest.mark.parametrize("has_q,has_v", [(False, False), (False, True), (True, False), (True, True)])
def test_legacy_fifteen_argument_packed_api_keeps_optional_lora_branches(has_q, has_v):
    from solution.unpad import RoPEPacked

    torch.manual_seed(843)
    inputs, metadata = _inputs()
    q, lq, k, v, lv = inputs
    selected = tuple(value for value, enabled in zip(inputs, (True, has_q, True, True, has_v))
                     if enabled)
    outputs = RoPEPacked.apply(q, lq if has_q else None, k, None, v,
                               lv if has_v else None, *metadata)
    gradients = torch.autograd.grad(outputs, selected, tuple(torch.randn_like(x) for x in outputs))
    assert len(gradients) == len(selected)
    assert all(value.dtype == torch.float16 for value in gradients)


@pytest.mark.parametrize("unsupported", ["nonhalf", "missing_q", "k_lora", "other_scale"])
def test_scaled_api_rejects_unsupported_inputs_before_rng_or_kernel_work(unsupported):
    from solution.unpad import RoPEPackedScaled

    inputs, metadata = _inputs()
    q, lq, k, v, lv = inputs
    lk, scale = None, 2.0
    if unsupported == "nonhalf":
        lq = lq.float()
    elif unsupported == "missing_q":
        lq = None
    elif unsupported == "k_lora":
        lk = k
    else:
        scale = float("inf")
    before = torch.cuda.get_rng_state()
    with pytest.raises(ValueError, match="private scaled packed RoPE"):
        RoPEPackedScaled.apply(q, lq, k, lk, v, lv, *metadata, scale, 2.0)
    assert torch.equal(torch.cuda.get_rng_state(), before)


@pytest.mark.parametrize("scale", [2, 0.0, 0.5, 65536.0, float("inf"), float("nan")])
def test_private_dispatch_rejects_other_scalars_and_missing_context(monkeypatch, scale):
    import solution.private_scope as private_scope
    from solution.unpad import _can_fuse_lora_scaling

    x = torch.randn((7, 64), device="cuda")
    step = SimpleNamespace(packed_metadata={})
    parts = (None, object(), object(), None, 2.0)
    monkeypatch.setattr(private_scope, "expected_loss_scale", lambda: 65536.0)
    with torch.autocast("cuda", dtype=torch.float16):
        assert _can_fuse_lora_scaling(parts, parts, x, step)
        invalid = (*parts[:4], scale)
        assert not _can_fuse_lora_scaling(invalid, parts, x, step)
        assert not _can_fuse_lora_scaling(parts, invalid, x, step)
        assert not _can_fuse_lora_scaling(parts, parts, x.half(), step)
        assert not _can_fuse_lora_scaling(parts, parts, x, SimpleNamespace(packed_metadata=None))
        assert not _can_fuse_lora_scaling((None, None, None, None, None), parts, x, step)
        assert not _can_fuse_lora_scaling(parts, (None, None, None, None, None), x, step)
        monkeypatch.setattr(private_scope, "expected_loss_scale", lambda: None)
        assert not _can_fuse_lora_scaling(parts, parts, x, step)
    assert not _can_fuse_lora_scaling(parts, parts, x, step)


def test_raw_lora_preserves_two_native_dropout_draws_and_linear_results():
    from solution.unpad import _Batch, _lora_out

    torch.manual_seed(844)
    batch, sequence, width = 2, 17, 32
    rows = torch.tensor([*range(17), *range(17, 26)], device="cuda", dtype=torch.int64)
    x = torch.randn((rows.numel(), width), device="cuda", requires_grad=True)
    parts = []
    for _ in range(2):
        parts.append((None, torch.nn.Linear(width, 16, bias=False).cuda(),
                      torch.nn.Linear(16, width, bias=False).cuda(), torch.nn.Dropout(0.05), 2.0))
    initial = torch.cuda.get_rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        reference_step = _Batch(rows, batch, sequence)
        expected = tuple(_lora_out(p, x, reference_step) for p in parts)
        expected_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(initial)
        actual_step = _Batch(rows, batch, sequence)
        raw = tuple(_lora_out(p, x, actual_step, apply_scale=False) for p in parts)
    assert torch.equal(torch.cuda.get_rng_state(), expected_state)
    for value, expected_value in zip(raw, expected):
        assert value.dtype == torch.float16
        _same_bits(value * 2.0, expected_value)


def test_unexpected_raw_dtype_uses_legacy_producer_without_replaying_projection(monkeypatch):
    import solution.unpad as unpad

    tokens, width = 3, 16
    x = torch.randn((tokens, width), device="cuda")
    weight = torch.empty((width, width), device="cuda", dtype=torch.float16)
    weight.quant_state = None
    base = SimpleNamespace(weight=weight)
    q_parts = (base, object(), object(), None, 2.0)
    k_parts = (base, None, None, None, None)
    v_parts = (base, object(), object(), None, 2.0)
    o_parts = (base, None, None, None, None)
    raw_q, raw_v = torch.ones_like(x), torch.full_like(x, 3.0)
    calls = []
    producers = []

    def projection(*args):
        calls.append("qkv")
        return tuple(torch.zeros_like(x, dtype=torch.float16) for _ in range(3))

    def lora(parts, values, step, *, apply_scale=True):
        calls.append(("q" if parts is q_parts else "v" if parts is v_parts
                      else "k" if parts is k_parts else "o", apply_scale))
        return raw_q if parts is q_parts else raw_v if parts is v_parts else None

    def legacy(q, lq, k, lk, v, lv, *args):
        producers.append("legacy")
        _same_bits(lq, raw_q * 2.0)
        _same_bits(lv, raw_v * 2.0)
        return tuple(torch.zeros((1, 1, tokens, width), device="cuda",
                                 dtype=torch.float16) for _ in range(3))

    def forbidden(*args):
        raise AssertionError("nonhalf output must keep the old packed producer")

    monkeypatch.setattr(unpad, "QKV4bit", SimpleNamespace(apply=projection))
    monkeypatch.setattr(unpad, "_lora_out", lora)
    monkeypatch.setattr(unpad, "_can_fuse_lora_scaling", lambda *args: True)
    monkeypatch.setattr(unpad, "RoPEPacked", SimpleNamespace(apply=legacy))
    monkeypatch.setattr(unpad, "RoPEPackedScaled", SimpleNamespace(apply=forbidden))
    monkeypatch.setattr(unpad, "MatMulNF4", SimpleNamespace(apply=lambda values, *args: values))
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, "default",
                        lambda q, k, v, *args, **kwargs: (q,))
    step = SimpleNamespace(rows=torch.arange(tokens, device="cuda"), batch=1,
                           seq_len=tokens, done_with=lambda value: None,
                           packed_metadata={"cumulative": torch.tensor([0, tokens], device="cuda",
                                                                        dtype=torch.int32),
                                            "maximum": tokens})
    meta = {"hq": 1, "hk": 1, "dim": width, "rep": 1}
    layer = SimpleNamespace(self_attn=SimpleNamespace(scaling=0.25))
    output = unpad._attention(layer, (q_parts, k_parts, v_parts, o_parts), x,
                              None, None, None, step, meta)
    assert output.shape == x.shape
    assert producers == ["legacy"]
    assert calls == ["qkv", ("q", False), ("k", True), ("v", False), ("o", True)]

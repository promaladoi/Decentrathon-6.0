"""Exact qualification of the private dropout/Linear-AMP cast fusion."""

from types import MethodType

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def _rows(batch, sequence, lengths):
    mask = torch.arange(sequence, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]
    return mask.reshape(-1).nonzero().squeeze(1)


def _bits(actual, expected):
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    integer = torch.int16 if expected.dtype == torch.float16 else torch.int32
    assert torch.equal(actual[finite].view(integer), expected[finite].view(integer))


@pytest.mark.parametrize("p", [.05, .021, .37])
@pytest.mark.parametrize("width", [31, 1536])
def test_half_store_matches_separate_cast_forward_backward_rng_and_saved_mask(p, width):
    from solution.dropout import real_row_dropout, real_row_dropout_linear_input

    torch.manual_seed(894)
    rows = _rows(2, 17, (17, 9))
    x = torch.randn((rows.numel(), width), device="cuda", requires_grad=True)
    upstream = torch.randn((width, rows.numel()), device="cuda", dtype=torch.float16).t()
    assert not upstream.is_contiguous()
    before = torch.cuda.get_rng_state()
    expected = real_row_dropout(x, rows, 2, 17, p, rows_from_nonzero=True).half()
    after = torch.cuda.get_rng_state()
    expected_next = torch.rand(29, device="cuda")
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    torch.cuda.set_rng_state(before)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value):
        actual = real_row_dropout_linear_input(x, rows, 2, 17, p, rows_from_nonzero=True)
    assert actual.dtype == torch.float16
    assert len(saved) == 1 and saved[0].dtype == torch.uint8
    assert saved[0].numel() == (x.numel() + 7) // 8
    assert torch.equal(torch.cuda.get_rng_state(), after)
    assert torch.equal(torch.rand(29, device="cuda"), expected_next)
    before_backward = torch.cuda.get_rng_state()
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    assert actual_dx.dtype == torch.float32
    assert torch.equal(torch.cuda.get_rng_state(), before_backward)
    _bits(actual, expected)
    _bits(actual_dx, expected_dx)


def test_exceptional_values_signed_zeros_and_retained_graph():
    from solution.dropout import reference_dropout, real_row_dropout_linear_input

    rows = _rows(2, 17, (17, 9))
    values = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0.,
                           1.e-8, -1.e-8, 1.5, -1.5, 65520.], device="cuda")
    width = 31
    x = values.repeat((rows.numel() * width + 9) // 10)[:rows.numel() * width].reshape(
        rows.numel(), width).clone().requires_grad_()
    upstream = values.flip(0).repeat((x.numel() + 9) // 10)[:x.numel()].reshape_as(x).half()
    torch.manual_seed(895)
    before = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, 2, 17, .37).half()
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    torch.cuda.set_rng_state(before)
    actual = real_row_dropout_linear_input(x, rows, 2, 17, .37, rows_from_nonzero=True)
    first_dx, = torch.autograd.grad(actual, x, upstream, retain_graph=True)
    second_dx, = torch.autograd.grad(actual, x, upstream)
    _bits(actual, expected)
    _bits(first_dx, expected_dx)
    _bits(second_dx, first_dx)


@pytest.mark.parametrize("sequence", [128, 384, 512])
def test_native_lora_projection_preserves_output_input_and_weight_gradients(monkeypatch, sequence):
    from solution import unpad

    torch.manual_seed(896)
    batch, width = 4, 1536
    rows = _rows(batch, sequence, (sequence, sequence // 2, sequence // 3, sequence // 5))
    x = torch.randn((rows.numel(), width), device="cuda", requires_grad=True)
    a = torch.nn.Linear(width, 16, bias=False, device="cuda")
    b = torch.nn.Linear(16, 256, bias=False, device="cuda")
    dropout = torch.nn.Dropout(.05).train()
    grad = torch.randn((rows.numel(), 256), device="cuda", dtype=torch.float16) * 16.
    original = unpad.real_row_dropout_linear_input
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(unpad, "real_row_dropout_linear_input", record)
    with torch.autocast("cuda", dtype=torch.float16):
        before = torch.cuda.get_rng_state()
        expected_step = unpad._Batch(rows, batch, sequence)
        expected = b(a(expected_step.dropout(dropout, x))) * 2.
        expected_step.done_with(x)
        after = torch.cuda.get_rng_state()
        expected_grads = torch.autograd.grad(expected, (x, a.weight, b.weight), grad)
        torch.cuda.set_rng_state(before)
        actual_step = unpad._Batch(rows, batch, sequence)
        actual = unpad._lora_out((None, a, b, dropout, 2.), x, actual_step)
        actual_step.done_with(x)
        assert torch.equal(torch.cuda.get_rng_state(), after)
        actual_grads = torch.autograd.grad(actual, (x, a.weight, b.weight), grad)
    _bits(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        _bits(actual_grad, expected_grad)
    assert len(calls) == (1 if sequence >= 384 else 0)


def test_generic_api_stays_fp32_even_under_half_autocast():
    from solution import unpad
    from solution.dropout import real_row_dropout

    rows = _rows(4, 384, (384, 192, 128, 76))
    x = torch.randn((rows.numel(), 1536), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        assert real_row_dropout(x, rows, 4, 384, rows_from_nonzero=True).dtype == torch.float32
        assert unpad._Batch(rows, 4, 384).dropout(torch.nn.Dropout(.05), x).dtype == torch.float32


@pytest.mark.parametrize("observed", ["custom_forward", "custom_call_impl", "class_call", "functional_linear",
                                       "local_pre", "local_post", "compiled", "accelerate", "global"])
def test_observable_or_custom_linear_rejects_private_cast(observed, monkeypatch):
    from solution import unpad

    rows = _rows(4, 384, (384, 192, 128, 76))
    x = torch.randn((rows.numel(), 1536), device="cuda")
    a = torch.nn.Linear(1536, 16, bias=False, device="cuda")
    handle = None
    if observed == "custom_forward":
        a.forward = MethodType(lambda self, value: torch.nn.functional.linear(value, self.weight), a)
    elif observed == "custom_call_impl":
        a._call_impl = MethodType(lambda self, *args, **kwargs: self.forward(*args, **kwargs), a)
    elif observed == "class_call":
        original = torch.nn.Linear.__call__
        monkeypatch.setattr(torch.nn.Linear, "__call__", lambda self, *args, **kwargs: original(self, *args, **kwargs))
    elif observed == "functional_linear":
        original = torch.nn.functional.linear
        monkeypatch.setattr(torch.nn.functional, "linear", lambda *args, **kwargs: original(*args, **kwargs))
    elif observed == "local_pre":
        handle = a.register_forward_pre_hook(lambda module, inputs: None)
    elif observed == "local_post":
        handle = a.register_forward_hook(lambda module, inputs, output: None)
    elif observed == "compiled":
        a._compiled_call_impl = lambda *args, **kwargs: None
    elif observed == "accelerate":
        a._hf_hook = object()
    else:
        handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda module, inputs: None)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            assert not unpad._can_cast_dropout_for_linear(a, x)
            actual = unpad._Batch(rows, 4, 384).dropout(torch.nn.Dropout(.05), x, linear_input=a)
            assert actual.dtype == torch.float32
    finally:
        if handle is not None:
            handle.remove()


def test_non_amp_keeps_original_fp32_dropout_dtype():
    from solution import unpad

    rows = _rows(4, 384, (384, 192, 128, 76))
    x = torch.randn((rows.numel(), 1536), device="cuda")
    a = torch.nn.Linear(1536, 16, bias=False, device="cuda")
    with torch.autocast("cuda", enabled=False):
        assert not unpad._can_cast_dropout_for_linear(a, x)
        actual = unpad._Batch(rows, 4, 384).dropout(torch.nn.Dropout(.05), x, linear_input=a)
        assert actual.dtype == torch.float32

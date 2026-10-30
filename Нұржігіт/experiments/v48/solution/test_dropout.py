"""Exact native-RNG and bit-packed saved-mask qualification."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _rows(batch, sequence, lengths):
    positions = torch.arange(sequence, device='cuda')[None]
    mask = positions < torch.tensor(lengths, device='cuda')[:, None]
    return mask.reshape(-1).nonzero().squeeze(1)


def _bits(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    # Include signed zero in finite-value equality; NaN payloads are not an API
    # contract and can change across the native index_add in the reference.
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))


@pytest.mark.parametrize('p', [0.05, 0.021, 0.37])
@pytest.mark.parametrize('shape,lengths', [((4, 128, 1536), (128, 91, 17, 83)),
                                         ((2, 17, 31), (17, 9))])
def test_exact_full_mask_real_output_gradient_and_rng_progression(p, shape, lengths):
    from solution.dropout import real_row_dropout

    batch, sequence, hidden = shape
    torch.manual_seed(610)
    rows = _rows(batch, sequence, lengths)
    x = torch.randn((rows.numel(), hidden), device='cuda', requires_grad=True)
    upstream = torch.randn_like(x) * 256.0
    state = torch.cuda.get_rng_state()
    padded = x.new_zeros((batch * sequence, hidden)).index_copy(0, rows, x).view(shape)
    full_output, full_mask = torch.ops.aten.native_dropout.default(padded, p, True)
    expected = full_output.view(batch * sequence, hidden).index_select(0, rows)
    expected_mask = full_mask.view(batch * sequence, hidden).index_select(0, rows)
    expected_rng = torch.cuda.get_rng_state()
    expected_next = torch.rand(41, device='cuda')
    expected_gradient, = torch.autograd.grad(expected, (x,), upstream)
    torch.cuda.set_rng_state(state)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        actual = real_row_dropout(x, rows, batch, sequence, p, rows_from_nonzero=True)
    assert len(saved) == 1 and saved[0].dtype == torch.uint8
    assert saved[0].shape == ((x.numel() + 7) // 8,)
    bit_index = torch.arange(x.numel(), device='cuda')
    unpacked = ((saved[0][bit_index // 8].to(torch.int32) >> (bit_index % 8)) & 1).bool()
    assert torch.equal(unpacked.reshape_as(expected_mask), expected_mask)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    assert torch.equal(torch.rand(41, device='cuda'), expected_next)
    state_before_backward = torch.cuda.get_rng_state()
    actual_gradient, = torch.autograd.grad(actual, (x,), upstream)
    assert torch.equal(torch.cuda.get_rng_state(), state_before_backward)
    _bits(actual, expected)
    _bits(actual_gradient, expected_gradient)


def test_two_shared_input_dropouts_preserve_order_and_gradient_join():
    from solution.dropout import real_row_dropout

    torch.manual_seed(611)
    batch, sequence, hidden = 4, 128, 1536
    rows = _rows(batch, sequence, (128, 39, 91, 12))
    x = torch.randn((rows.numel(), hidden), device='cuda', requires_grad=True)
    gq, gv = torch.randn_like(x), torch.randn_like(x)
    padded = x.new_zeros((batch * sequence, hidden)).index_copy(0, rows, x).view(
        batch, sequence, hidden)
    initial = torch.cuda.get_rng_state()
    q = F.dropout(padded, p=0.05, training=True).view(-1, hidden).index_select(0, rows)
    v = F.dropout(padded, p=0.05, training=True).view(-1, hidden).index_select(0, rows)
    after_reference = torch.cuda.get_rng_state()
    expected_dx, = torch.autograd.grad((q, v), (x,), (gq, gv))
    torch.cuda.set_rng_state(initial)
    aq = real_row_dropout(x, rows, batch, sequence, rows_from_nonzero=True)
    av = real_row_dropout(x, rows, batch, sequence, rows_from_nonzero=True)
    assert torch.equal(torch.cuda.get_rng_state(), after_reference)
    actual_dx, = torch.autograd.grad((aq, av), (x,), (gq, gv))
    _bits(aq, q)
    _bits(av, v)
    _bits(actual_dx, expected_dx)


def test_exceptional_values_and_retained_graph_use_native_multiply_semantics():
    from solution.dropout import real_row_dropout, reference_dropout

    torch.manual_seed(612)
    batch, sequence, hidden = 2, 17, 64
    rows = _rows(batch, sequence, (17, 11))
    values = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0, -0.0, 2.0, -2.0],
                          device='cuda')
    x = values.repeat((rows.numel() * hidden + 6) // 7)[:rows.numel() * hidden].reshape(
        rows.numel(), hidden).clone().requires_grad_()
    upstream = values.flip(0).repeat((x.numel() + 6) // 7)[:x.numel()].reshape_as(x)
    state = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, batch, sequence)
    expected_dx, = torch.autograd.grad(expected, (x,), upstream)
    torch.cuda.set_rng_state(state)
    actual = real_row_dropout(x, rows, batch, sequence, rows_from_nonzero=True)
    first_dx, = torch.autograd.grad(actual, (x,), upstream, retain_graph=True)
    second_dx, = torch.autograd.grad(actual, (x,), upstream)
    _bits(actual, expected)
    _bits(first_dx, expected_dx)
    _bits(second_dx, first_dx)


@pytest.mark.parametrize('p,training', [(0.0, True), (1.0, True), (0.05, False)])
def test_nonrandom_shortcuts_and_untrusted_rows_use_reference(p, training):
    from solution.dropout import real_row_dropout, reference_dropout

    rows = _rows(2, 17, (17, 9))
    x = torch.randn((rows.numel(), 64), device='cuda', requires_grad=True)
    upstream = torch.randn_like(x)
    state = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, 2, 17, p, training)
    expected_dx, = torch.autograd.grad(expected, (x,), upstream)
    torch.cuda.set_rng_state(state)
    actual = real_row_dropout(x, rows, 2, 17, p, training, rows_from_nonzero=True)
    actual_dx, = torch.autograd.grad(actual, (x,), upstream)
    _bits(actual, expected)
    _bits(actual_dx, expected_dx)


def test_untrusted_rows_default_keeps_reference_rng_and_graph():
    from solution.dropout import real_row_dropout, reference_dropout

    rows = _rows(2, 17, (17, 9))
    x = torch.randn((rows.numel(), 64), device='cuda', requires_grad=True)
    state = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, 2, 17)
    after_reference = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    actual = real_row_dropout(x, rows, 2, 17)
    _bits(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), after_reference)


def test_compile_only_warmup_does_not_advance_rng():
    from solution.dropout import warmup
    state = torch.cuda.get_rng_state()
    warmup()
    assert torch.equal(torch.cuda.get_rng_state(), state)

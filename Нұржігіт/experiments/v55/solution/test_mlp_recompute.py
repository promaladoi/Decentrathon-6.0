"""Exact activation-recompute qualification, independent of contest metrics."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA GPU')


def _layers(hidden, inter):
    from solution.test_layer import _to_4bit
    return tuple(_to_4bit(torch.nn.Linear(a, b, bias=False))
                 for a, b in ((hidden, inter), (hidden, inter), (inter, hidden)))


def _reference(x, gate, up, down):
    # Independent autograd branches preserve the two half gradients' FP32
    # addition, as in native Linear4bit (one cast for each projection).
    from solution.nf4 import MatMulNF4
    from solution.swiglu import swiglu
    g = MatMulNF4.apply(x.half(), gate.weight, None, gate.weight.quant_state)
    u = MatMulNF4.apply(x.half(), up.weight, None, up.weight.quant_state)
    return MatMulNF4.apply(swiglu(g, u), down.weight, None, down.weight.quant_state)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('rows,hidden,inter', [(31, 64, 192), (128, 1536, 8960), (832, 1536, 8960)])
def test_recompute_exact_values_gradients_rng_and_retained_graph(dtype, rows, hidden, inter):
    from solution.mlp import MLP4bit
    torch.manual_seed(5501)
    layers = _layers(hidden, inter)
    x = torch.randn(rows, hidden, device='cuda', dtype=dtype, requires_grad=True)
    grad = torch.randn(rows, hidden, device='cuda', dtype=torch.float16)
    reference = _reference(x, *layers)
    expected_dx, = torch.autograd.grad(reference, (x,), grad)
    before = torch.cuda.get_rng_state().clone()
    actual = MLP4bit.apply(x, *layers, torch.float16)
    gate, narrow_input = actual.grad_fn.saved_tensors
    assert gate.shape == (rows, inter) and narrow_input.shape == x.shape
    assert gate.dtype == narrow_input.dtype == torch.float16
    assert gate.numel() + narrow_input.numel() < 2*rows*inter
    first_dx, = torch.autograd.grad(actual, (x,), grad, retain_graph=True)
    second_dx, = torch.autograd.grad(actual, (x,), grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(first_dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(second_dx, first_dx, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), before)


@pytest.mark.parametrize('mutate', ['packed', 'scales', 'metadata'])
def test_recompute_rejects_changed_frozen_up(mutate):
    from solution.mlp import MLP4bit
    layers = _layers(64, 192)
    x = torch.randn(31, 64, device='cuda', requires_grad=True)
    out = MLP4bit.apply(x, *layers, torch.float16)
    state = layers[1].weight.quant_state
    with torch.no_grad():
        if mutate == 'packed': layers[1].weight.add_(0)
        elif mutate == 'scales': state.absmax.add_(0)
        else: state.blocksize *= 2
    with pytest.raises(RuntimeError, match='frozen up projection changed'):
        out.sum().backward()


def test_half_input_mutation_does_not_change_saved_forward_value():
    from solution.mlp import MLP4bit
    layers = _layers(64, 192)
    x = torch.randn(31, 64, device='cuda', dtype=torch.float16, requires_grad=True)
    actual = MLP4bit.apply(x, *layers, torch.float16)
    grad = torch.randn_like(actual)
    before, = torch.autograd.grad(actual, (x,), grad, retain_graph=True)
    with torch.no_grad(): x.add_(1)
    after, = torch.autograd.grad(actual, (x,), grad)
    torch.testing.assert_close(after, before, rtol=0, atol=0)

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
    narrow_input, = actual.grad_fn.saved_tensors
    assert narrow_input.shape == x.shape
    assert narrow_input.dtype == torch.float16
    assert narrow_input.numel() < 2*rows*inter
    first_dx, = torch.autograd.grad(actual, (x,), grad, retain_graph=True)
    second_dx, = torch.autograd.grad(actual, (x,), grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(first_dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(second_dx, first_dx, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), before)


@pytest.mark.parametrize('mutate', ['packed', 'scales', 'metadata',
                                  'scale_replacement', 'storage_replacement'])
@pytest.mark.parametrize('projection', [0, 1])
def test_recompute_rejects_changed_frozen_projection(mutate, projection):
    from solution.mlp import MLP4bit
    layers = _layers(64, 192)
    x = torch.randn(31, 64, device='cuda', requires_grad=True)
    out = MLP4bit.apply(x, *layers, torch.float16)
    state = layers[projection].weight.quant_state
    with torch.no_grad():
        if mutate == 'packed': layers[projection].weight.add_(0)
        elif mutate == 'scales': state.absmax.add_(0)
        elif mutate == 'metadata': state.blocksize *= 2
        elif mutate == 'scale_replacement': state.absmax = state.absmax.clone()
        else: layers[projection].weight.data = layers[projection].weight.data.clone()
    with pytest.raises(RuntimeError, match='frozen (gate|up) projection changed'):
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


def test_saved_activation_hooks_can_offload_and_restore_retained_graph():
    from solution.mlp import MLP4bit
    layers = _layers(64, 192)
    x = torch.randn(31, 64, device='cuda', requires_grad=True)
    upstream = torch.randn(31, 64, device='cuda', dtype=torch.float16)
    reference = _reference(x, *layers)
    reference_dx, = torch.autograd.grad(reference, (x,), upstream)
    packed_shapes = []
    def pack(tensor):
        packed_shapes.append(tuple(tensor.shape))
        return tensor.device, tensor.cpu()
    def unpack(saved):
        device, tensor = saved
        return tensor.to(device)
    before = torch.cuda.get_rng_state().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = MLP4bit.apply(x, *layers, torch.float16)
    assert packed_shapes == [(31, 64)]
    first_dx, = torch.autograd.grad(out, (x,), upstream, retain_graph=True)
    second_dx, = torch.autograd.grad(out, (x,), upstream)
    for actual in (first_dx, second_dx):
        torch.testing.assert_close(actual, reference_dx, rtol=0, atol=0)
    assert torch.equal(before, torch.cuda.get_rng_state())

"""Local backward reuse preserves native dispatch, bytes, gradients and lifetime."""
from collections import Counter
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA NF4')


def _cases(hidden, inter):
    from .test_mlp_recompute import _layers
    from .test_cpu_streaming import _offload
    native = _layers(hidden, inter)
    cpu = []
    for layer in native:
        packed, state = _offload(layer)
        weight = torch.nn.Parameter(packed, requires_grad=False)
        weight.quant_state = state
        cpu.append(SimpleNamespace(weight=weight))
    return native, tuple(cpu)


def test_mismatched_state_dtype_keeps_the_existing_forward(monkeypatch):
    from . import mlp
    layer = SimpleNamespace(weight=SimpleNamespace(quant_state=SimpleNamespace(dtype=torch.float32)))
    sentinel, seen = object(), []
    def forward(*args):
        seen.append(args)
        return sentinel
    def forbidden(*args):
        raise AssertionError('dtype mismatch must not predecode')
    monkeypatch.setattr(mlp, '_linear_forward', forward)
    monkeypatch.setattr(mlp, '_weight', forbidden)
    output, weight = mlp._recompute_projection(None, layer, torch.float16)
    assert output is sentinel and weight is None and len(seen) == 1


@cuda
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('rows,hidden,inter', [(257, 128, 256), (832, 1536, 8960)])
def test_real_cpu_weights_reuse_once_exactly_and_never_live_in_context(monkeypatch, dtype, rows, hidden, inter):
    from . import mlp, nf4, cpu_streaming
    from .test_mlp_recompute import _reference
    torch.manual_seed(8021)
    native, cpu = _cases(hidden, inter)
    x = torch.randn((rows, hidden), device='cuda', dtype=dtype, requires_grad=True)
    upstream = torch.randn((rows, hidden), device='cuda', dtype=torch.float16)
    reference = _reference(x, *native)
    expected_dx, = torch.autograd.grad(reference, x, upstream)
    actual = mlp.MLP4bit.apply(x, *cpu, torch.float16)
    assert not nf4._uses_native_custom_gemm(x.half(), cpu[0].weight.quant_state)
    assert not nf4._uses_native_custom_gemm(x.half(), cpu[1].weight.quant_state)
    saved, = actual.grad_fn.saved_tensors
    assert saved.shape == x.shape and saved.dtype == torch.float16
    assert 'gate_weight' not in vars(actual.grad_fn)
    assert 'up_weight' not in vars(actual.grad_fn)
    transfers, decodes, live_weights = Counter(), [], []
    native_transfer = cpu_streaming.transfer_nf4
    native_decode = nf4._decode_uncached
    native_weight = mlp._weight
    def transfer(packed, state, device=None):
        transfers[id(packed)] += 1
        return native_transfer(packed, state, device)
    def decode(packed, state, dtype):
        decodes.append(tuple(state.shape))
        return native_decode(packed, state, dtype)
    def weight(layer, dtype):
        result = native_weight(layer, dtype)
        if layer is cpu[0] or layer is cpu[1]:
            live_weights.append(weakref.ref(result))
        return result
    monkeypatch.setattr(cpu_streaming, 'transfer_nf4', transfer)
    monkeypatch.setattr(nf4, '_decode_uncached', decode)
    monkeypatch.setattr(mlp, '_weight', weight)
    rng = torch.cuda.get_rng_state().clone()
    for repetition in range(2):
        transfers.clear()
        decodes.clear()
        dx, = torch.autograd.grad(actual, x, upstream, retain_graph=repetition == 0)
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        assert transfers == Counter({id(layer.weight): 1 for layer in cpu})
        assert len(decodes) == 3, 'gate/up/down each decode exactly once per backward'
        gc.collect()
        assert all(reference() is None for reference in live_weights)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@cuda
def test_real_native_custom_gemm_is_not_replaced_by_dense_recompute(monkeypatch):
    from . import mlp, nf4, cpu_streaming
    from .test_mlp_recompute import _reference
    torch.manual_seed(8022)
    native, cpu = _cases(1536, 256)
    x = torch.randn((1, 1536), device='cuda', dtype=torch.float16, requires_grad=True)
    if not all(nf4._uses_native_custom_gemm(x, layer.weight.quant_state) for layer in cpu[:2]):
        pytest.skip('native custom NF4 GEMM not selected for this device/shape')
    expected = _reference(x, *native)
    upstream = torch.randn_like(expected)
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    actual = mlp.MLP4bit.apply(x, *cpu, torch.float16)
    transfers, recomputes, decodes = Counter(), [], []
    native_transfer = cpu_streaming.transfer_nf4
    native_forward = mlp._linear_forward
    native_decode = nf4._decode_uncached
    def transfer(packed, state, device=None):
        transfers[id(packed)] += 1
        return native_transfer(packed, state, device)
    def forward(value, packed, bias, state):
        recomputes.append(id(packed))
        return native_forward(value, packed, bias, state)
    def decode(packed, state, dtype):
        decodes.append(tuple(state.shape))
        return native_decode(packed, state, dtype)
    monkeypatch.setattr(cpu_streaming, 'transfer_nf4', transfer)
    monkeypatch.setattr(mlp, '_linear_forward', forward)
    monkeypatch.setattr(nf4, '_decode_uncached', decode)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    assert recomputes == [id(cpu[0].weight), id(cpu[1].weight)]
    assert transfers[id(cpu[0].weight)] == transfers[id(cpu[1].weight)] == 2
    assert transfers[id(cpu[2].weight)] == 1
    assert len(decodes) == 3, 'custom forward has no decode; only three input-gradient decodes'


@cuda
@pytest.mark.parametrize('projection', [0, 1])
@pytest.mark.parametrize('mutation', ['packed', 'scale', 'replacement'])
def test_cpu_weight_mutation_after_forward_is_still_rejected(projection, mutation):
    from .mlp import MLP4bit
    _, layers = _cases(128, 256)
    x = torch.randn((257, 128), device='cuda', requires_grad=True)
    out = MLP4bit.apply(x, *layers, torch.float16)
    weight = layers[projection].weight
    if mutation == 'packed':
        weight.view(-1)[0] ^= 1
    elif mutation == 'scale':
        weight.quant_state.absmax.view(-1)[0] ^= 1
    else:
        weight.data = weight.data.clone()
    with pytest.raises(RuntimeError, match='frozen (gate|up) projection changed'):
        out.sum().backward()

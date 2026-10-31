"""GPU correctness of exact CPU NF4 byte streaming and tied CPU lookup."""
import copy

import pytest
import torch

from .cpu_streaming import (_embedding_forward, _head_forward, _pin_state,
                            streamed, transfer_nf4)
from .nf4 import MatMulNF4, _linear_forward, dequantize_nf4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _offload(layer):
    state = copy.deepcopy(layer.weight.quant_state)
    state.to(torch.device('cpu'))
    _pin_state(state)
    state._solution_cpu_streamed = True
    state._solution_execution_device = torch.device('cuda', torch.cuda.current_device())
    return layer.weight.detach().cpu().pin_memory(), state


@pytest.mark.parametrize('rows', [1, 32, 257])
@pytest.mark.parametrize('nested', [False, True])
def test_cpu_nf4_stream_matches_cuda_values_gradient_and_rng(rows, nested):
    import bitsandbytes as bnb
    torch.manual_seed(71)
    layer = bnb.nn.Linear4bit(128, 256, bias=False, compute_dtype=torch.float16,
                             compress_statistics=nested, quant_type='nf4').cuda()
    packed, state = _offload(layer)
    assert streamed(packed, state)
    x = torch.randn(rows, 128, device='cuda', dtype=torch.float16, requires_grad=True)
    rng = torch.cuda.get_rng_state().clone()
    reference = MatMulNF4.apply(x, layer.weight, None, layer.weight.quant_state)
    actual = MatMulNF4.apply(x, packed, None, state)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual_dx, = torch.autograd.grad(actual, x, gradient, retain_graph=True)
    reference_dx, = torch.autograd.grad(reference, x, gradient)
    torch.testing.assert_close(actual_dx, reference_dx, rtol=0, atol=0)
    again_dx, = torch.autograd.grad(actual, x, gradient)
    torch.testing.assert_close(again_dx, reference_dx, rtol=0, atol=0)
    # Only the explicit generation of gradient above may consume RNG.
    after = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(rng)
    torch.randn_like(actual)
    assert torch.equal(torch.cuda.get_rng_state(), after)
    assert packed.device.type == 'cpu' and state.code.device.type == 'cpu'
    assert layer.weight.is_cuda and layer.weight.quant_state.code.is_cuda


def test_independent_transfer_does_not_cache_stale_bytes_or_mutate_metadata():
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(128, 256, bias=False, compute_dtype=torch.float16,
                             compress_statistics=True, quant_type='nf4').cuda()
    packed, state = _offload(layer)
    live, live_state = transfer_nf4(packed, state)
    before = live.clone()
    packed.view(-1)[0] ^= 1
    later, later_state = transfer_nf4(packed, state)
    assert torch.equal(live, before)
    assert not torch.equal(later, before)
    assert live_state is not later_state and live_state is not state
    assert state.code.device.type == 'cpu'
    assert live_state.code.is_cuda and later_state.code.is_cuda
    assert state.state2.absmax.device.type == 'cpu'
    assert state.state2.code.device.type == 'cpu'
    assert live_state.state2 is not state.state2
    assert live_state.state2 is not later_state.state2
    assert live_state.state2.code.is_cuda and later_state.state2.code.is_cuda


def test_cpu_embedding_and_head_preserve_original_values_and_tie():
    from .frozen_storage import install_frozen_storage
    from types import MethodType
    embedding = torch.nn.Embedding(2048, 32)
    embedding.weight.data = embedding.weight.data.half().float()
    embedding.weight.requires_grad_(False)
    head = torch.nn.Linear(32, 2048, bias=False)
    head.weight = embedding.weight
    ids = torch.tensor([1, 5, 17], device='cuda')
    reference = embedding(ids.cpu()).cuda()
    original = embedding.weight
    state = install_frozen_storage(embedding, head)
    assert state is not None
    embedding.__dict__['_solution_original_embedding_dtype'] = torch.float32
    embedding.forward = MethodType(_embedding_forward, embedding)
    head.forward = MethodType(_head_forward, head)
    actual = embedding(ids)
    assert embedding.weight is head.weight is original
    assert actual.dtype == torch.float32 and actual.is_cuda
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    x = torch.randn(3, 32, device='cuda', dtype=torch.float32, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.float16):
        expected = torch.nn.functional.linear(x, original.detach().cuda())
        actual_logits = head(x)
    torch.testing.assert_close(actual_logits, expected, rtol=0, atol=0)
    dx, = torch.autograd.grad(actual_logits.sum(), x)
    expected_dx, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)


def test_generic_cpu_tensor_is_not_treated_as_owned_streaming_weight():
    from types import SimpleNamespace
    assert not streamed(torch.zeros(16, dtype=torch.uint8), SimpleNamespace())


@pytest.mark.parametrize('rows', [129, 513])
@pytest.mark.parametrize('scale', [1024.0, 32.0])
@pytest.mark.parametrize('ignored', [False, True])
def test_cpu_head_offload_same_chunk_exact_and_native_gradient_qualified(monkeypatch, scale, ignored, rows):
    """Offload is exact at equal M; smaller cuBLAS M has a separate bound.

    Existing head qualification uses <=0.2% gradient relative L2 against
    native half Linear -> float CE. The absolute check also scales with the
    half output precision and the maximum frozen weight, not a relative
    elementwise bound that becomes meaningless at near-zero derivatives.
    """
    from . import lmhead
    from .test_lmhead_scaled import _native, _assert_gradient
    monkeypatch.setattr(lmhead, '_private_expected_loss_scale', lambda: 1024.0)
    torch.manual_seed(1213 + rows)
    cpu_weight = torch.nn.Parameter(torch.randn(2048, 32).half(), requires_grad=False)
    cpu_weight._solution_cpu_streamed = True
    gpu_weight = cpu_weight.detach().cuda()
    x = torch.randn(rows, 32, device='cuda', dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 2048, (rows,), device='cuda')
    if ignored:
        targets.fill_(-100)
    rng = torch.cuda.get_rng_state().clone()
    # Explicit independent CUDA-weight execution at the same token split:
    # no CPU weight, transfer, CPU saved tensors or CPU backward state.
    with monkeypatch.context() as matching:
        matching.setattr(lmhead, 'CHUNK', 256)
        reference, reference_stats = lmhead.lm_head_loss(x, gpu_weight, targets, rows)
    baseline512, _ = lmhead.lm_head_loss(x, gpu_weight, targets, rows)
    dense_reference = _native(x, gpu_weight, targets, rows)
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        actual, actual_stats = lmhead.lm_head_loss(x, cpu_weight, targets, rows)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual, baseline512, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual, dense_reference, rtol=1e-4, atol=1e-5)
    for key in ('entropy', 'argmax', 'ce'):
        torch.testing.assert_close(actual_stats[key], reference_stats[key], rtol=0, atol=0)
    gradient = torch.tensor(scale, device='cuda')
    dx, = torch.autograd.grad(actual, x, gradient, retain_graph=True)
    expected_dx, = torch.autograd.grad(reference, x, gradient)
    baseline_dx, = torch.autograd.grad(baseline512, x, gradient)
    dense_dx, = torch.autograd.grad(dense_reference, x, gradient)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    repeated, = torch.autograd.grad(actual, x, gradient)
    torch.testing.assert_close(repeated, expected_dx, rtol=0, atol=0)
    absolute_bound = (4 * torch.finfo(torch.float16).eps * abs(scale) / rows
                      * gpu_weight.abs().max().float()).item()
    diagnostics = []
    for name, native in [('native512', baseline_dx), ('native_dense', dense_dx)]:
        _assert_gradient(dx, native, targets == -100)
        error = (dx.float() - native.float())
        maximum = error.abs().max().item()
        relative_l2 = (error.norm() / native.float().norm().clamp(min=1e-12)).item()
        assert maximum <= absolute_bound
        diagnostics.append(f'{name}:max_abs={maximum:.9g},relative_l2={relative_l2:.9g}')
    print(f'CPU_HEAD_QUALIFIED rows={rows} scale={scale} ignored={ignored} '
          f'absolute_bound={absolute_bound:.9g} ' + ' '.join(diagnostics))
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert cpu_weight.device.type == 'cpu'

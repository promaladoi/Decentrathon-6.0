"""Lossless offload, original mutation versions, streams and gradient reuse."""
import gc
import weakref

import pytest
import torch

from solution.activation_storage import ActivationStorage, _Kept, _Offloaded


def test_cpu_version_witness_retains_counter_without_original_storage():
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    manager = ActivationStorage()
    witness = manager._witness(x[:, 1:])
    assert witness.device.type == 'cpu' and witness.numel() == 0
    assert witness.untyped_storage().nbytes() == 0
    version = witness._version
    x.add_(1)
    assert witness._version == version + 1


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA offload')


@cuda
@pytest.mark.parametrize('shape,stride', [((64, 96), (96, 1)), ((96, 64), (1, 96)),
                                        ((96, 64), (128, 1))])
def test_offload_exact_values_shape_strides_and_shared_versions(shape, stride):
    tensor = torch.empty_strided(shape, stride, device='cuda', dtype=torch.float16)
    tensor.copy_(torch.arange(tensor.numel(), device='cuda').reshape(shape))
    manager = ActivationStorage(minimum_bytes=0)
    packed = manager.pack(tensor)
    assert isinstance(packed, _Offloaded)
    assert packed.host.device.type == 'cpu' and packed.host.is_pinned()
    assert packed.witness.device.type == 'cpu' and packed.witness.numel() == 0
    restored = manager.unpack(packed)
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
    assert restored.stride() == tensor.stride()
    restored.add_(1)
    with pytest.raises(RuntimeError, match='modified inplace'):
        manager.unpack(packed)


@cuda
def test_original_tensor_may_die_but_alias_mutation_is_detected():
    base = torch.randn(192, 192, device='cuda')
    view = base[:, 32:160]
    ref = weakref.ref(view)
    manager = ActivationStorage(minimum_bytes=0)
    packed = manager.pack(view)
    del view
    gc.collect()
    assert ref() is None
    base.add_(1)
    with pytest.raises(RuntimeError, match='modified inplace'):
        manager.unpack(packed)


@cuda
def test_saved_activation_gpu_allocation_is_released():
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    tensor = torch.ones((1024, 1024), device='cuda', dtype=torch.float32)
    ref = weakref.ref(tensor)
    manager = ActivationStorage(minimum_bytes=0)
    packed = manager.pack(tensor)
    del tensor
    gc.collect()
    torch.cuda.synchronize()
    assert ref() is None
    assert packed.witness.untyped_storage().nbytes() == 0
    assert torch.cuda.memory_allocated() <= before + 4096
    torch.testing.assert_close(manager.unpack(packed),
                               torch.ones((1024, 1024), device='cuda'), rtol=0, atol=0)


@cuda
def test_parameter_storage_is_kept_and_mutation_guarded():
    model = torch.nn.Linear(256, 256, bias=False, device='cuda')
    manager = ActivationStorage(model, minimum_bytes=0)
    packed = manager.pack(model.weight.t())
    assert isinstance(packed, _Kept)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match='modified inplace'):
        manager.unpack(packed)


@cuda
def test_gradient_rng_and_two_backward_reuses_equal_ordinary():
    a = torch.randn(128, 96, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    weight = torch.randn(96, 48, device='cuda', requires_grad=True)
    other_weight = weight.detach().clone().requires_grad_()
    torch.manual_seed(4531)
    reference = torch.nn.functional.dropout(a, 0.05, True).matmul(weight).sin().sum()
    expected_rng = torch.cuda.get_rng_state()
    torch.manual_seed(4531)
    manager = ActivationStorage(minimum_bytes=0)
    with manager.hooks():
        actual = torch.nn.functional.dropout(b, 0.05, True).matmul(other_weight).sin().sum()
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for _ in range(2):
        expected_grads = torch.autograd.grad(reference, (a, weight), retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (b, other_weight), retain_graph=True)
        for expected, value in zip(expected_grads, actual_grads):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert manager.stats['offloaded_tensors'] > 0


@cuda
def test_cross_stream_restore_waits_for_the_pinned_snapshot():
    tensor = torch.arange(1024 * 1024, device='cuda', dtype=torch.float32)
    producer = torch.cuda.current_stream()
    manager = ActivationStorage(minimum_bytes=0)
    packed = manager.pack(tensor)
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        restored = manager.unpack(packed)
        result = restored.sum()
    consumer.synchronize()
    torch.testing.assert_close(result, tensor.sum(), rtol=0, atol=0)
    producer.synchronize()


@cuda
def test_duplicate_save_reuses_one_host_copy():
    tensor = torch.ones((256, 256), device='cuda')
    manager = ActivationStorage(minimum_bytes=0)
    first, second = manager.pack(tensor), manager.pack(tensor)
    assert first is second
    assert manager.stats['offloaded_tensors'] == 1
    assert manager.stats['reused_copies'] == 1


@cuda
def test_gapped_saved_view_preserves_backward_layout_and_gradient():
    class SaveGapped(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            view = value[:, 16:112]
            ctx.save_for_backward(view)
            ctx.shape, ctx.strides = value.shape, view.stride()
            return view.square().sum()

        @staticmethod
        def backward(ctx, upstream):
            view, = ctx.saved_tensors
            assert view.stride() == ctx.strides
            grad = torch.zeros(ctx.shape, device=view.device, dtype=view.dtype)
            grad[:, 16:112] = 2 * view * upstream
            return grad

    reference = torch.randn(128, 192, device='cuda', requires_grad=True)
    candidate = reference.detach().clone().requires_grad_()
    expected = SaveGapped.apply(reference)
    manager = ActivationStorage(minimum_bytes=0)
    with manager.hooks():
        actual = SaveGapped.apply(candidate)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(candidate.grad, reference.grad, rtol=0, atol=0)


@cuda
def test_overlapping_or_pathological_spans_keep_the_original_tensor():
    manager = ActivationStorage(minimum_bytes=0)
    base = torch.ones((32, 32), device='cuda')
    overlapping = base[:1].expand(32, 32)
    pathological = base.as_strided((2,), (512,))
    assert isinstance(manager.pack(overlapping), _Kept)
    assert isinstance(manager.pack(pathological), _Kept)

"""Frozen-head cast caching keeps source identity, values and old graphs."""

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def _empty_head_cache():
    from solution.head_weight_cache import clear_half_weight_cache

    clear_half_weight_cache()
    yield
    clear_half_weight_cache()


def _weight(device="cpu"):
    return torch.nn.Parameter(torch.randn((19, 16), device=device), requires_grad=False)


def test_repeated_frozen_weight_cast_reuses_copy_without_mutating_model_or_rng():
    from solution.head_weight_cache import get_half_weight

    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(19, 16)
    model.head = torch.nn.Linear(16, 19, bias=False)
    model.head.weight = model.embedding.weight
    model.requires_grad_(False)
    weight = model.head.weight
    before = weight.detach().clone()
    version, pointer = weight._version, weight.data_ptr()
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state()
    first = get_half_weight(weight)
    second = get_half_weight(weight)
    assert first is second and first.dtype == torch.float16
    assert first is not weight and first.data_ptr() != pointer
    torch.testing.assert_close(first, before.half(), rtol=0, atol=0)
    assert model.embedding.weight is weight and model.head.weight is weight
    assert weight.dtype == torch.float32 and not weight.requires_grad
    assert (weight._version, weight.data_ptr()) == (version, pointer)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    assert tuple(model.state_dict()) == keys
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("mutation", ["add", "copy", "set", "load_state"])
def test_normal_weight_mutation_invalidates_without_overwriting_old_copy(mutation):
    from solution.head_weight_cache import get_half_weight

    model = torch.nn.Linear(16, 19, bias=False).requires_grad_(False)
    weight = model.weight
    old = get_half_weight(weight)
    old_values = old.clone()
    replacement = torch.full_like(weight, 0.125)
    if mutation == "load_state":
        model.load_state_dict({"weight": replacement})
    else:
        with torch.no_grad():
            if mutation == "add":
                weight.add_(0.125)
            elif mutation == "copy":
                weight.copy_(replacement)
            else:
                weight.set_(replacement)
    new = get_half_weight(weight)
    assert new is not old and get_half_weight(weight) is new
    torch.testing.assert_close(new, weight.half(), rtol=0, atol=0)
    torch.testing.assert_close(old, old_values, rtol=0, atol=0)


def test_replacement_and_explicit_clear_create_new_copies():
    from solution.head_weight_cache import get_half_weight, clear_half_weight_cache

    first_weight = _weight()
    second_weight = torch.nn.Parameter(first_weight.detach().clone(), requires_grad=False)
    first = get_half_weight(first_weight)
    second = get_half_weight(second_weight)
    assert second is not first
    clear_half_weight_cache(first_weight)  # does not evict a different source
    assert get_half_weight(second_weight) is second
    clear_half_weight_cache(second_weight)
    assert get_half_weight(second_weight) is not second


@pytest.mark.parametrize("mutation", ["value", "requires_grad"])
def test_mutated_exposed_half_copy_is_not_reused(mutation):
    from solution.head_weight_cache import get_half_weight

    weight = _weight()
    old = get_half_weight(weight)
    if mutation == "value":
        old.add_(1)
    else:
        old.requires_grad_(True)
    new = get_half_weight(weight)
    assert new is not old and not new.requires_grad
    torch.testing.assert_close(new, weight.half(), rtol=0, atol=0)


def test_single_slot_and_weak_source_do_not_keep_dead_models_or_old_casts_alive():
    from solution.head_weight_cache import get_half_weight

    first_weight = _weight()
    first = get_half_weight(first_weight)
    old_copy = weakref.ref(first)
    del first
    second_weight = _weight()
    second = get_half_weight(second_weight)
    gc.collect()
    assert old_copy() is None
    source, half = weakref.ref(second_weight), weakref.ref(second)
    del second_weight, second
    gc.collect()
    assert source() is None and half() is None


def test_noncontiguous_offset_source_and_metadata_changes_match_native_cast():
    from solution.head_weight_cache import get_half_weight

    base = torch.randn((23, 20))
    weight = base[2:21, 2:18].t()
    assert not weight.is_contiguous() and weight.storage_offset() != 0
    half = get_half_weight(weight)
    assert get_half_weight(weight) is half
    torch.testing.assert_close(half, weight.half(), rtol=0, atol=0)
    with torch.no_grad():
        weight.set_(torch.randn((16, 19)))
    new = get_half_weight(weight)
    assert new is not half
    torch.testing.assert_close(new, weight.half(), rtol=0, atol=0)


def test_inference_context_does_not_create_an_inference_cached_tensor():
    from solution.head_weight_cache import get_half_weight

    weight = _weight()
    with torch.inference_mode():
        half = get_half_weight(weight)
    assert not half.is_inference()
    x = torch.randn((3, 16), requires_grad=True)
    F.linear(x.half(), half).float().sum().backward()
    assert x.grad is not None


def test_trainable_input_keeps_autograd_and_old_graphs_keep_old_weight_copy():
    from solution.head_weight_cache import get_half_weight

    weight = _weight()
    original_values = weight.detach().clone()
    x = torch.randn((3, 16), requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    half = get_half_weight(weight)
    loss = F.linear(x.half(), half).float().square().sum()
    expected = F.linear(reference.half(), original_values.half()).float().square().sum()
    with torch.no_grad():
        weight.add_(0.25)
    assert get_half_weight(weight) is not half
    first, = torch.autograd.grad(loss, x, retain_graph=True)
    second, = torch.autograd.grad(loss, x)
    expected_dx, = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(first, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_dx, rtol=0, atol=0)
    weight.requires_grad_(True)
    trainable_half = get_half_weight(weight)
    assert trainable_half.requires_grad and trainable_half.grad_fn is not None
    trainable_half.float().sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)


def test_unsupported_weak_reference_uses_ordinary_cast(monkeypatch):
    import solution.head_weight_cache as cache

    def no_weak_reference(*_args):
        raise TypeError("not weak-referenceable")
    monkeypatch.setattr(cache, "_weak_reference", no_weak_reference)
    weight = _weight()
    first, second = cache.get_half_weight(weight), cache.get_half_weight(weight)
    assert first is not second
    torch.testing.assert_close(first, weight.half(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_cache_preserves_values_rng_and_uses_fresh_copy_on_another_stream():
    from solution.head_weight_cache import get_half_weight

    weight = _weight("cuda")
    state = torch.cuda.get_rng_state()
    default = torch.cuda.current_stream()
    first = get_half_weight(weight)
    stream = torch.cuda.Stream()
    stream.wait_stream(default)
    with torch.cuda.stream(stream):
        second = get_half_weight(weight)
        assert second is not first
        assert get_half_weight(weight) is second
    default.wait_stream(stream)
    torch.testing.assert_close(first, weight.half(), rtol=0, atol=0)
    torch.testing.assert_close(second, weight.half(), rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), state)

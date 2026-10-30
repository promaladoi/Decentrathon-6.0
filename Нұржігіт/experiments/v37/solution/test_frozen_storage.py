"""Exact contracts and conservative fallback for frozen tied storage.

CPU tests exercise the module contract. CUDA cases additionally validate the
actual packing kernel and fp16 autocast; only the contest harness measures speed.
"""
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from solution.frozen_storage import exact_half_copy, install_frozen_storage


def pair(device='cpu'):
    embedding = nn.Embedding(1024, 64, device=device)
    head = nn.Linear(64, 1024, bias=False, device=device)
    # The actual model's fp16 -> fp32 preparation, with exactly representable values.
    embedding.weight = nn.Parameter(embedding.weight.half().float(), requires_grad=False)
    head.weight = embedding.weight
    return embedding, head


def test_lossless_storage_ties_keys_and_embedding_dtype():
    embedding, head = pair()
    weight = head.weight
    before = weight.detach().clone()
    keys = (tuple(embedding.state_dict()), tuple(head.state_dict()))
    ids = torch.tensor([0, 19, 1023])
    expected = embedding(ids)
    state = install_frozen_storage(embedding, head)
    assert state['storage_bytes_saved'] == before.numel() * 2
    assert head.weight is embedding.weight is weight
    assert head.weight.dtype == torch.float16 and not head.weight.requires_grad
    assert keys == (tuple(embedding.state_dict()), tuple(head.state_dict()))
    assert torch.equal(head.weight.float(), before)
    actual = embedding(ids)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('autocast', [False, True])
def test_forward_and_input_gradient_exact(autocast):
    embedding, head = pair()
    x = torch.randn(5, 64, requires_grad=True)
    context = lambda: torch.autocast('cpu', dtype=torch.bfloat16) if autocast else nullcontext()
    with context():
        expected = head(x)
    g = torch.randn_like(expected)
    expected_grad, = torch.autograd.grad(expected, x, g)
    assert install_frozen_storage(embedding, head) is not None
    with context():
        actual = head(x)
    actual_grad, = torch.autograd.grad(actual, x, g)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize('value', [1.00001, float('inf'), float('nan')])
def test_inexact_or_nonfinite_values_leave_original_storage(value):
    embedding, head = pair()
    embedding.weight.data[17, 5] = value
    ptr = head.weight.data_ptr()
    assert install_frozen_storage(embedding, head) is None
    assert head.weight.dtype == torch.float32 and head.weight.data_ptr() == ptr
    assert '_solution_frozen_storage' not in head.__dict__


@pytest.mark.parametrize('case', ['trainable', 'untied', 'bias', 'max_norm', 'hook', 'forward'])
def test_unsupported_pairs_are_unchanged(case):
    embedding, head = pair()
    handle = None
    if case == 'trainable':
        head.weight.requires_grad_(True)
    elif case == 'untied':
        head.weight = nn.Parameter(head.weight.clone(), requires_grad=False)
    elif case == 'bias':
        head.bias = nn.Parameter(torch.zeros(1024), requires_grad=False)
    elif case == 'max_norm':
        embedding.max_norm = 1.0
    elif case == 'hook':
        handle = embedding.register_forward_hook(lambda *_: None)
    else:
        head.forward = lambda x: x
    assert install_frozen_storage(embedding, head) is None
    assert embedding.weight.dtype == head.weight.dtype == torch.float32
    if handle is not None:
        handle.remove()


def test_install_is_idempotent_and_preserves_parameter():
    embedding, head = pair()
    first = install_frozen_storage(embedding, head)
    ptr = head.weight.data_ptr()
    assert install_frozen_storage(embedding, head) is first
    assert head.weight.data_ptr() == ptr


def test_unfreezing_restores_fp32_before_gradients():
    embedding, head = pair()
    state = install_frozen_storage(embedding, head)
    head.weight.requires_grad_(True)
    head(torch.randn(3, 64)).sum().backward()
    assert not state['active']
    assert head.weight is embedding.weight
    assert head.weight.dtype == head.weight.grad.dtype == torch.float32


def test_replaced_head_weight_is_used_without_stale_cache():
    embedding, head = pair()
    install_frozen_storage(embedding, head)
    head.weight = nn.Parameter(torch.randn(1024, 64), requires_grad=False)
    x = torch.randn(3, 64)
    torch.testing.assert_close(head(x), nn.functional.linear(x, head.weight), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_autocast_forward_and_input_gradients_exact():
    embedding, head = pair('cuda')
    x = torch.randn(5, 64, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.float16):
        expected = head(x)
    g = torch.randn_like(expected)
    expected_grad, = torch.autograd.grad(expected, x, g)
    assert install_frozen_storage(embedding, head) is not None
    with torch.autocast('cuda', dtype=torch.float16):
        actual = head(x)
    actual_grad, = torch.autograd.grad(actual, x, g)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('value', [0.125, 1.00001, float('inf'), float('nan')])
def test_cuda_value_proof_kernel_and_tail(value):
    weight = torch.randn(4097, device='cuda').half().float()
    weight[-1] = value
    packed = exact_half_copy(weight)
    if value == 0.125:
        assert packed is not None
        torch.testing.assert_close(packed.float(), weight, rtol=0, atol=0)
    else:
        assert packed is None

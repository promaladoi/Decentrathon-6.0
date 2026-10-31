"""Temporary logits lifetime, without pretending CPU mocks test Triton math.

CPU tests execute the actual _stats function from lmhead.py, with just the GPU
kernel launch replaced by a no-retention stub. This tests Python ownership and
unchanged GEMM scheduling. CUDA tests use the real full head and its kernels.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch


class _NoRetentionKernel:
    def __getitem__(self, grid):
        def launch(logits, targets, rows, entropy, argmax, ce, lse, *args, **kwargs):
            entropy.index_fill_(0, rows, 0)
            argmax.index_fill_(0, rows, 0)
            if ce is not entropy:
                ce.index_fill_(0, rows, 0)
                lse.zero_()
        return launch


def _python_stats(namespace):
    path = Path(__file__).with_name('lmhead.py')
    tree = ast.parse(path.read_bytes())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_stats']
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_stats']


def _exercise_python_stats(mode, function_loader=_python_stats):
    """Use CPU tensors and weak refs; never retain the observed GEMM outputs."""
    rows, hidden, vocabulary, chunk_size = 11, 8, 97, 3
    x, weight = torch.randn(rows, hidden), torch.randn(vocabulary, hidden)
    ids = torch.arange(rows)
    targets = torch.zeros(rows, dtype=torch.long)
    entropy, ce = torch.empty(rows), torch.zeros(rows)
    argmax = torch.empty(rows, dtype=torch.long)
    references, observations, sizes = [], [], []

    def mm(left, right):
        assert right.shape == (hidden, vocabulary)
        observations.append(sum(ref() is not None for ref in references))
        value = torch.mm(left, right)
        references.append(weakref.ref(value))
        sizes.append(value.shape[0])
        return value

    def gradient(logits, lse, selected_targets, scale, w16, overwrite=False):
        assert overwrite
        return torch.mm(logits, w16)

    namespace = {
        'torch': SimpleNamespace(mm=mm, empty=torch.empty, float32=torch.float32),
        'CHUNK': chunk_size, 'BLOCK_V': 4096,
        'lm_ce_scatter_kernel': _NoRetentionKernel(), '_gradient_mm': gradient,
    }
    stats = function_loader(namespace)
    saved, lses = [], []
    hidden_gradient = torch.zeros_like(x)
    stats(x, weight, ids, targets, entropy, argmax, None if mode == 'prompt' else ce,
          saved_chunks=saved if mode == 'saved' else None,
          precompute=(torch.tensor(1.0), hidden_gradient, lses) if mode == 'private' else None)
    return observations, sizes, references, saved, lses


@pytest.mark.parametrize('mode', ['prompt', 'private'])
def test_python_temporary_chunks_die_before_next_gemm(mode):
    observations, sizes, refs, saved, lses = _exercise_python_stats(mode)
    assert sizes == [3, 3, 3, 2]
    assert observations == [0, 0, 0, 0]
    assert all(ref() is None for ref in refs)
    assert not saved
    assert [item.numel() for item in lses] == ([3, 3, 3, 2] if mode == 'private' else [])


def test_general_backward_keeps_every_original_chunk():
    observations, sizes, refs, saved, lses = _exercise_python_stats('saved')
    assert sizes == [3, 3, 3, 2]
    assert observations == [0, 1, 2, 3]
    assert not lses
    assert len(saved) == 4
    assert all(ref() is pair[0] for ref, pair in zip(refs, saved))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton head')
@pytest.mark.parametrize('expected_scale,actual_scale', [(None, 1.0), (1024.0, 1024.0), (1024.0, 1.0)])
def test_cuda_head_lifetime_and_retained_graph(monkeypatch, expected_scale, actual_scale):
    import solution.lmhead as head
    from solution.test_lmhead_scaled import _assert_gradient, _native
    monkeypatch.setattr(head, 'CHUNK', 3)
    monkeypatch.setattr(head, '_private_expected_loss_scale', lambda: expected_scale)
    torch.manual_seed(445)
    x = torch.randn(13, 32, device='cuda', requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4103, 32, device='cuda') * .05
    targets = torch.randint(4103, (13,), device='cuda')
    targets[::3] = -100
    expected = _native(reference, weight, targets, None)
    refs, alive = [], []
    original_mm = torch.mm

    def record(left, right):
        is_head = tuple(right.shape) == (32, 4103)
        if is_head:
            alive.append(sum(ref() is not None for ref in refs))
        result = original_mm(left, right)
        if is_head:
            refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch, 'mm', record)
    loss, _ = head.lm_head_loss(x, weight, targets, None)
    torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-5)
    if expected_scale is not None:
        assert alive and all(n == 0 for n in alive)
        assert all(ref() is None for ref in refs)
    else:
        labelled_chunks = (int((targets != -100).sum()) + 2) // 3
        assert sum(ref() is not None for ref in refs) == labelled_chunks
    upstream = torch.tensor(actual_scale, device='cuda')
    expected.backward(upstream)
    loss.backward(upstream, retain_graph=True)
    _assert_gradient(x.grad, reference.grad, targets == -100)
    first = x.grad.detach().clone()
    x.grad = None
    loss.backward(upstream)
    torch.testing.assert_close(x.grad, first, rtol=0, atol=0)
    if expected_scale is not None:
        assert all(n == 0 for n in alive)

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('active', ['all', 'some', 'none'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_sparse_backward_matches_dense(active, dtype):
    from solution.frozen_head import frozen_head
    torch.manual_seed(53)
    x = torch.randn(2, 7, 128, device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(1031, 128, device='cuda', dtype=torch.float16) * .03
    expected = torch.nn.functional.linear(x.half(), w)
    actual = frozen_head(x, w)
    grad = torch.randn_like(expected)
    if active == 'some': grad[:, ::2] = 0
    if active == 'none': grad.zero_()
    eg, = torch.autograd.grad(expected, x, grad)
    ag, = torch.autograd.grad(actual, x, grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ag, eg, rtol=2e-3, atol=2e-4)
    if active != 'all': assert torch.count_nonzero(ag[:, ::2]).item() == 0


def test_cache_invalidation_and_trainable_fallback():
    from solution.frozen_head import install_head
    layer = torch.nn.Linear(128, 1031, bias=False, device='cuda')
    layer.weight.requires_grad_(False)
    x = torch.randn(2, 5, 128, device='cuda', requires_grad=True)
    original = layer.forward
    before_keys = list(layer.state_dict())
    install_head(layer)
    with torch.autocast('cuda', dtype=torch.float16):
        torch.testing.assert_close(layer(x), original(x), rtol=0, atol=0)
        first = layer._team_half_cache[2]
        layer(x)
        assert layer._team_half_cache[2] is first
        with torch.no_grad(): layer.weight.add_(.01)
        torch.testing.assert_close(layer(x), original(x), rtol=0, atol=0)
        assert layer._team_half_cache[2] is not first
        layer.weight.requires_grad_(True)
        layer(x).float().sum().backward()
        assert layer.weight.grad is not None
        assert not hasattr(layer, '_team_half_cache')
    assert list(layer.state_dict()) == before_keys

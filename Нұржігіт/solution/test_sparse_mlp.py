import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('active', ['all', 'some', 'none'])
def test_sparse_mlp_matches_dense_forward_and_backward(active):
    import bitsandbytes as bnb
    from solution.sparse_mlp import sparse_mlp
    from solution.swiglu import swiglu
    torch.manual_seed(59)
    layers = []
    for i, o in [(128, 256), (128, 256), (256, 128)]:
        layer = bnb.nn.Linear4bit(i, o, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type='nf4')
        layer = layer.to(dtype=torch.float16).cuda()
        layer.weight.requires_grad_(False)
        layers.append(layer)
    gate, up, down = layers
    x = torch.randn(2, 7, 128, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.float16):
        expected = down(swiglu(gate(x.half()), up(x.half()))).float()
        actual = sparse_mlp(x, gate, up, down)
    grad = torch.randn_like(expected)
    if active == 'some': grad[:, ::2] = 0
    if active == 'none': grad.zero_()
    eg, = torch.autograd.grad(expected, x, grad)
    ag, = torch.autograd.grad(actual, x, grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ag, eg, rtol=3e-3, atol=3e-4)
    if active != 'all': assert torch.count_nonzero(ag[:, ::2]).item() == 0

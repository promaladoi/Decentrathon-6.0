import pytest
import torch
import torch.nn.functional as F
from solution.trim_attention import occupied_spans, trimmed_sdpa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('full', [False, True])
def test_trimmed_sdpa_values_and_gradients(left, full):
    torch.manual_seed(83)
    batch, heads, length, dim = 4, 4, 128, 64
    lengths = [128]*4 if full else [128, 73, 25, 11]
    occupied = torch.zeros(batch, length, device='cuda', dtype=torch.long)
    for i, n in enumerate(lengths):
        if left:
            occupied[i, length-n:] = 1
        else:
            occupied[i, :n] = 1
    causal = torch.ones(length, length, device='cuda', dtype=torch.bool).tril()
    mask = causal[None, None] & occupied[:, None, None, :].bool()
    inputs = [torch.randn(batch, heads, length, dim, device='cuda', dtype=torch.float16,
                          requires_grad=True) for _ in range(3)]
    scale = dim**-.5
    ref = F.scaled_dot_product_attention(*inputs, attn_mask=mask, scale=scale)
    got = trimmed_sdpa(*inputs, mask, occupied_spans(occupied), scale=scale)
    real = occupied[:, None, :, None].bool().expand_as(ref)
    torch.testing.assert_close(got[real], ref[real], rtol=3e-3, atol=3e-3)
    grad = torch.randn_like(ref) * real
    expected = torch.autograd.grad(ref, inputs, grad, retain_graph=True)
    actual = torch.autograd.grad(got, inputs, grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=5e-3, atol=5e-3)

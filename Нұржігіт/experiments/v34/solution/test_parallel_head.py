import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('vocab', [10007, 151936])
@pytest.mark.parametrize('with_grad', [False, True])
def test_parallel_statistics_and_gradient(vocab, with_grad):
    from solution.parallel_head import reduce_chunk
    torch.manual_seed(71)
    logits = torch.randn(19, vocab, device='cuda', dtype=torch.float16)
    # Equal maxima across tiles must pick the first global index.
    logits[0, 7] = 20
    logits[0, vocab - 1] = 20
    ref = logits.float().requires_grad_()
    target = torch.randint(vocab, (19,), device='cuda')
    ce = torch.nn.functional.cross_entropy(ref, target, reduction='none')
    prob = ref.softmax(-1)
    entropy = -(prob * ref.log_softmax(-1)).sum(-1)
    expected_grad, = torch.autograd.grad(ce.sum() * 1024, ref)
    e, a, c = reduce_chunk(logits, target if with_grad else None)
    torch.testing.assert_close(e, entropy.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(a, ref.argmax(-1), rtol=0, atol=0)
    if with_grad:
        torch.testing.assert_close(c, ce.detach(), rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(logits, expected_grad.half(), rtol=2e-3, atol=1e-3)


def test_combined_chunked_head_loss_and_hidden_gradient():
    from solution.parallel_head import patch_parallel_head
    from solution.team_vladimir.test_unpad import test_lm_head_loss_matches_cross_entropy
    patch_parallel_head()
    for with_num_items in (True, False):
        test_lm_head_loss_matches_cross_entropy(with_num_items)


def test_current_head_and_dropout_warmup_signatures_compile():
    # Exercise the actual production signatures; do not swallow a failed
    # warmup as the startup thread deliberately does for its safe fallback.
    from solution.parallel_head import warmup
    warmup()

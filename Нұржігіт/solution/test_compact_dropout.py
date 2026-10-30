import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('p', [0.05, 0.25])
def test_same_dropout_values_gradients_and_rng(dtype, p):
    from solution.compact_dropout import compact_dropout
    batch, seq_len, hidden = 4, 127, 1536
    rows = torch.arange(batch * seq_len, device='cuda')[::3]
    x = torch.randn(rows.numel(), hidden, device='cuda', dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    torch.cuda.manual_seed(101)
    before = torch.cuda.get_rng_state()
    padded = torch.zeros(batch * seq_len, hidden, device='cuda', dtype=dtype)
    padded = padded.index_copy(0, rows, x).view(batch, seq_len, hidden)
    expected = torch.nn.functional.dropout(padded, p=p, training=True).view(-1, hidden)[rows]
    after = torch.cuda.get_rng_state()
    eg, = torch.autograd.grad(expected, x, grad)
    torch.cuda.set_rng_state(before)
    actual = compact_dropout(x, rows, batch, seq_len, p)
    ag, = torch.autograd.grad(actual, x, grad)
    assert torch.equal(torch.cuda.get_rng_state(), after)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ag, eg, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_successive_dropout_calls_preserve_rng_sequence(dtype):
    from solution.compact_dropout import compact_dropout
    batch, seq_len, hidden = 4, 131, 128
    rows = torch.arange(batch * seq_len, device='cuda')[::2]
    x = torch.randn(rows.numel(), hidden, device='cuda', dtype=dtype)
    torch.cuda.manual_seed(113)
    start = torch.cuda.get_rng_state()
    padded = torch.zeros(batch * seq_len, hidden, device='cuda', dtype=dtype)
    padded = padded.index_copy(0, rows, x).view(batch, seq_len, hidden)
    expected = [torch.nn.functional.dropout(padded, p=.05, training=True)
                .view(-1, hidden)[rows] for _ in range(2)]
    end = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(start)
    actual = [compact_dropout(x, rows, batch, seq_len, .05) for _ in range(2)]
    assert torch.equal(torch.cuda.get_rng_state(), end)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

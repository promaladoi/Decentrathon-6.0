import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('n', [2048, 2045])
def test_all_packed_bytes_tail_and_write_guards(dtype, n):
    from solution.compact_dropout import compact_dropout_pack_rows
    byte_values = torch.arange(256, device='cuda', dtype=torch.int32)
    bits = ((byte_values[:, None] >> torch.arange(8, device='cuda')) & 1).bool()
    rows = torch.arange(256, device='cuda')
    x = torch.ones(n, device='cuda', dtype=dtype)
    out = torch.empty_like(x)
    size = (n + 7) // 8
    packed = torch.full((size + 4,), 204, dtype=torch.uint8, device='cuda')
    compact_dropout_pack_rows[((n + 1023) // 1024,)](
        x, bits, rows, out, packed, n, 8, 1.0, 1024,
        num_warps=4, enable_fp_fusion=False)
    expected_bytes = byte_values.to(torch.uint8)
    if n % 8:
        expected_bytes[-1] &= (1 << (n % 8)) - 1
    assert torch.equal(packed[:size], expected_bytes)
    assert (packed[size:] == 204).all().item()
    torch.testing.assert_close(out, bits.flatten()[:n].to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('hidden', [7, 17, 65])
def test_packed_tail_keeps_values_gradients_and_rng(dtype, hidden):
    from solution.compact_dropout import compact_dropout
    batch, seq_len = 2, 7
    rows = torch.arange(batch * seq_len, device='cuda')[::3]
    x = torch.randn(rows.numel(), hidden, dtype=dtype, device='cuda', requires_grad=True)
    grad = torch.randn_like(x)
    torch.cuda.manual_seed(197)
    start = torch.cuda.get_rng_state()
    padded = torch.zeros(batch * seq_len, hidden, dtype=dtype, device='cuda')
    expected = torch.nn.functional.dropout(padded.index_copy(0, rows, x).view(batch, seq_len, hidden),
                                            .25, training=True).view(-1, hidden)[rows]
    end = torch.cuda.get_rng_state()
    expected_grad, = torch.autograd.grad(expected, x, grad)
    torch.cuda.set_rng_state(start)
    actual = compact_dropout(x, rows, batch, seq_len, .25)
    actual_grad, = torch.autograd.grad(actual, x, grad)
    assert torch.equal(torch.cuda.get_rng_state(), end)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('seq_len', [23, 127, 512])
def test_native_producer_preserves_full_mask_and_rng(dtype, seq_len):
    from solution.compact_dropout import _draw_full_shape_mask
    batch, hidden = 3, 1536
    ones = torch.ones(batch, seq_len, hidden, dtype=dtype, device='cuda')
    torch.cuda.manual_seed(193)
    start = torch.cuda.get_rng_state()
    expected = torch.nn.functional.dropout(ones, .05, training=True).ne(0)
    end = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(start)
    actual = _draw_full_shape_mask(ones.view(-1, hidden), batch, seq_len, .05)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), end)


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


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_index_storage_layout_does_not_change_dropout(dtype):
    from solution.compact_dropout import compact_dropout
    batch, seq_len, hidden = 3, 23, 128
    strided = torch.arange(batch * seq_len, device='cuda')[::3]
    contiguous = strided.contiguous()
    assert strided.stride(0) == 3 and contiguous.stride(0) == 1
    x = torch.randn(strided.numel(), hidden, device='cuda', dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    torch.cuda.manual_seed(127)
    state = torch.cuda.get_rng_state()
    a = compact_dropout(x, strided, batch, seq_len, .05)
    ga, = torch.autograd.grad(a, x, grad)
    end = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    b = compact_dropout(x, contiguous, batch, seq_len, .05)
    gb, = torch.autograd.grad(b, x, grad)
    assert torch.equal(torch.cuda.get_rng_state(), end)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)

"""Independent byte-layout, bounds and noncontiguous-backward checks."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _pack_reference(mask):
    flat = mask.flatten().to(torch.int64)
    padded = torch.zeros(((flat.numel() + 7) // 8) * 8, device=flat.device, dtype=torch.int64)
    padded[:flat.numel()] = flat
    shifts = torch.arange(8, device=flat.device, dtype=torch.int64)
    return (padded.reshape(-1, 8) << shifts).sum(1).to(torch.uint8)


@pytest.mark.parametrize('width,count', [(8, 256), (1, 7), (7, 147), (9, 115), (31, 67)])
def test_packed_gather_all_byte_patterns_row_crossings_and_guards(width, count):
    from solution.dropout import BLOCK, real_dropout_gather_kernel
    import triton

    full_rows = count + 2
    # For width eight this independently enumerates all 256 byte patterns.
    position = torch.arange(full_rows * width, device='cuda')
    full_mask = (((position // 8) >> (position % 8)) & 1).bool().reshape(full_rows, width)
    rows = torch.arange(count - 1, -1, -1, device='cuda', dtype=torch.int64)
    x = torch.randn((count, width), device='cuda')
    n = x.numel()
    expected_mask = full_mask.index_select(0, rows)
    expected = _pack_reference(expected_mask)
    storage = torch.full((expected.numel() + 2,), 173, device='cuda', dtype=torch.uint8)
    output_storage = torch.full((n + 2,), 123.0, device='cuda')
    packed = storage[1:-1]
    output = output_storage[1:-1].reshape_as(x)
    real_dropout_gather_kernel[(triton.cdiv(n, BLOCK),)](
        x, full_mask, rows, output, packed, n, full_rows, width, 1.0, BLOCK,
        num_warps=4, enable_fp_fusion=False)
    assert torch.equal(packed, expected)
    assert torch.equal(storage[[0, -1]], torch.tensor([173, 173], device='cuda', dtype=torch.uint8))
    assert torch.equal(output_storage[[0, -1]], torch.tensor([123.0, 123.0], device='cuda'))
    torch.testing.assert_close(output, x * expected_mask, rtol=0, atol=0)
    if n % 8:
        assert int(packed[-1]) >> (n % 8) == 0


@pytest.mark.parametrize('n', [1, 7, 8, 9, 1023, 1024, 1025])
def test_unpack_backward_partial_bytes_tile_boundaries_and_guards(n):
    from solution.dropout import BLOCK, real_dropout_backward_kernel
    import triton

    values = torch.tensor([0.0, -0.0, float('nan'), float('inf'), -float('inf'), 1.25, -3.0],
                          device='cuda')
    grad = values.repeat((n + 6) // 7)[:n].clone()
    mask = (torch.arange(n, device='cuda') % 3 != 0)
    packed = _pack_reference(mask)
    storage = torch.full((n + 2,), 123.0, device='cuda')
    real_dropout_backward_kernel[(triton.cdiv(n, BLOCK),)](
        grad, packed, storage[1:-1], n, 2.0, BLOCK, num_warps=4, enable_fp_fusion=False)
    expected = mask.float() * torch.where(grad == 0, 0.0, grad) * 2.0
    actual = storage[1:-1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
    assert torch.equal(storage[[0, -1]], torch.tensor([123.0, 123.0], device='cuda'))


def test_noncontiguous_upstream_preserves_native_gradient_and_rng():
    from solution.dropout import real_row_dropout, reference_dropout

    batch, sequence, width = 2, 17, 31
    mask = torch.arange(sequence, device='cuda')[None] < torch.tensor([17, 9], device='cuda')[:, None]
    rows = mask.reshape(-1).nonzero().squeeze(1)
    x = torch.randn((rows.numel(), width), device='cuda', requires_grad=True)
    upstream = torch.randn((width, rows.numel()), device='cuda').t()
    assert not upstream.is_contiguous()
    initial = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, batch, sequence, p=0.37)
    expected_rng = torch.cuda.get_rng_state()
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    torch.cuda.set_rng_state(initial)
    actual = real_row_dropout(x, rows, batch, sequence, p=0.37, rows_from_nonzero=True)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)

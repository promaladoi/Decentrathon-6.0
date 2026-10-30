"""Compare fused producer cast to native FP32 dropout followed by FP16 cast."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _bits(actual, expected):
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual[finite].view(integer), expected[finite].view(integer))


def _rows(batch, sequence, lengths):
    valid = torch.arange(sequence, device='cuda')[None] < torch.tensor(lengths, device='cuda')[:, None]
    return valid.flatten().nonzero().squeeze(1)


@pytest.mark.parametrize('p', [.05, .021, .37])
@pytest.mark.parametrize('shape,lengths', [((4, 512, 1536), (512, 173, 284, 97)),
                                         ((2, 17, 31), (17, 9))])
def test_half_producer_exact_rng_values_fp32_backward_and_retained_graph(p, shape, lengths):
    from solution.dropout import real_row_dropout, reference_dropout
    torch.manual_seed(4603)
    batch, sequence, width = shape
    rows = _rows(batch, sequence, lengths)
    x = torch.randn((rows.numel(), width), device='cuda', requires_grad=True)
    upstream = torch.randn((width, rows.numel()), device='cuda', dtype=torch.float16).T
    initial = torch.cuda.get_rng_state()
    expected = reference_dropout(x, rows, batch, sequence, p).half()
    end = torch.cuda.get_rng_state()
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    torch.cuda.set_rng_state(initial)
    actual = real_row_dropout(x, rows, batch, sequence, p,
                              rows_from_nonzero=True, output_dtype=torch.float16)
    assert torch.equal(torch.cuda.get_rng_state(), end)
    dx1, = torch.autograd.grad(actual, x, upstream, retain_graph=True)
    dx2, = torch.autograd.grad(actual, x, upstream)
    assert dx1.dtype == torch.float32
    _bits(actual, expected)
    _bits(dx1, expected_dx)
    _bits(dx2, expected_dx)


def test_two_lora_branches_exact_parameter_gradients_and_join():
    from solution.dropout import reference_dropout
    from solution.unpad import _Batch, _lora_out
    torch.manual_seed(4604)
    rows = _rows(4, 512, (512, 173, 284, 97))
    x = torch.randn((rows.numel(), 1536), device='cuda', requires_grad=True)
    branches = [(None, torch.nn.Linear(1536, 16, bias=False).cuda(),
                  torch.nn.Linear(16, 256, bias=False).cuda(), torch.nn.Dropout(.05), 2.0)
                for _ in range(2)]
    params = [x] + [m.weight for branch in branches for m in branch[1:3]]
    upstream = [torch.randn((rows.numel(), 256), device='cuda', dtype=torch.float16) for _ in branches]
    initial = torch.cuda.get_rng_state()
    with torch.autocast('cuda', dtype=torch.float16):
        expected = [b(a(reference_dropout(x, rows, 4, 512, .05))) * scale
                    for _, a, b, drop, scale in branches]
    end = torch.cuda.get_rng_state()
    expected_grads = torch.autograd.grad(expected, params, upstream)
    torch.cuda.set_rng_state(initial)
    step = _Batch(rows, 4, 512, packed_metadata={})
    with torch.autocast('cuda', dtype=torch.float16):
        actual = [_lora_out(branch, x, step) for branch in branches]
    assert torch.equal(torch.cuda.get_rng_state(), end)
    actual_grads = torch.autograd.grad(actual, params, upstream)
    for a, b in zip(actual + list(actual_grads), expected + list(expected_grads)):
        _bits(a, b)


def test_observed_linear_keeps_fp32_input():
    from solution.unpad import _Batch, _lora_out
    rows = _rows(4, 512, (512, 173, 284, 97))
    x = torch.randn((rows.numel(), 1536), device='cuda')
    a = torch.nn.Linear(1536, 16, bias=False).cuda()
    b = torch.nn.Linear(16, 256, bias=False).cuda()
    observed = []
    handle = a.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].dtype))
    try:
        with torch.autocast('cuda', dtype=torch.float16):
            _lora_out((None, a, b, torch.nn.Dropout(.05), 2.0), x,
                      _Batch(rows, 4, 512, packed_metadata={}))
    finally:
        handle.remove()
    assert observed == [torch.float32]


def test_replaced_linear_forward_keeps_fp32_input(monkeypatch):
    from solution.unpad import _Batch, _lora_out
    rows = _rows(4, 512, (512, 173, 284, 97))
    x = torch.randn((rows.numel(), 1536), device='cuda')
    a = torch.nn.Linear(1536, 16, bias=False).cuda()
    b = torch.nn.Linear(16, 256, bias=False).cuda()
    native = torch.nn.Linear.forward
    observed = []
    def replacement(module, inputs):
        if module is a:
            observed.append(inputs.dtype)
        return native(module, inputs)
    monkeypatch.setattr(torch.nn.Linear, 'forward', replacement)
    with torch.autocast('cuda', dtype=torch.float16):
        _lora_out((None, a, b, torch.nn.Dropout(.05), 2.0), x,
                  _Batch(rows, 4, 512, packed_metadata={}))
    assert observed == [torch.float32]

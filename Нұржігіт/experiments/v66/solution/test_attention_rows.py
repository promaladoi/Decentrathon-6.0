"""GPU qualification of isolated native attention alternatives; no installation."""
import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Kaggle T4 CUDA')


@CUDA
@pytest.mark.parametrize('lengths', [(31,17,9,3),(16,16,16,16),(19,11,7,1)])
@pytest.mark.parametrize('variant', ['equalheads','rows','packed','causal_bias'])
def test_outputs_and_real_token_qkv_gradients(lengths, variant):
    from solution.attention_rows import make_layout, VARIANTS
    torch.manual_seed(4411)
    batch, heads, kv_heads, dim = len(lengths), 12, 2, 128
    sequence = max(lengths)
    layout = make_layout(lengths, sequence, 'cuda')
    q = (torch.randn(batch, heads, sequence, dim, device='cuda',dtype=torch.float16)*0.2).requires_grad_()
    k = (torch.randn(batch, kv_heads, sequence, dim, device='cuda',dtype=torch.float16)*0.2).requires_grad_()
    v = (torch.randn_like(k)*0.2).requires_grad_()
    upstream = torch.randn(layout.total, heads*dim, device='cuda',dtype=torch.float16)
    scale = dim**-0.5
    def run(name):
        # Recreate this graph each time; all variants start from identical leaves.
        repeated = name != 'current' or not layout.all_full
        kk = k.repeat_interleave(heads//kv_heads,dim=1) if repeated else k
        vv = v.repeat_interleave(heads//kv_heads,dim=1) if repeated else v
        out = VARIANTS[name](q,kk,vv,layout,scale)
        grads = torch.autograd.grad(out,(q,k,v),upstream)
        return out,grads
    expected, reference_grads = run('current')
    actual, grads = run(variant)
    torch.testing.assert_close(actual,expected,rtol=3e-3,atol=3e-4)
    for result, reference in zip(grads,reference_grads):
        relative=(result.float()-reference.float()).norm()/reference.float().norm().clamp(min=1e-8)
        assert torch.isfinite(relative) and float(relative)<1e-2
        for b,n in enumerate(lengths):
            assert torch.count_nonzero(result[b,:,n:])==0


@CUDA
def test_per_row_auto_and_forced_native_efficient_match():
    from solution.attention_rows import make_layout,per_row_causal
    torch.manual_seed(4412)
    q,k,v=[torch.randn(4,12,29,128,device='cuda',dtype=torch.float16) for _ in range(3)]
    layout=make_layout([29,21,9,2],29,'cuda')
    a=per_row_causal(q,k,v,layout)
    b=per_row_causal(q,k,v,layout,force_efficient=True)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@CUDA
def test_packed_native_retained_graph_and_no_rng_draw():
    from solution.attention_rows import make_layout,packed_from_padded
    q,k,v=[torch.randn(4,12,17,128,device='cuda',dtype=torch.float16,requires_grad=True)
           for _ in range(3)]
    layout=make_layout([17,12,8,1],17,'cuda')
    state=torch.cuda.get_rng_state().clone()
    out=packed_from_padded(q,k,v,layout)
    assert torch.equal(state,torch.cuda.get_rng_state())
    grad=torch.randn_like(out)
    a=torch.autograd.grad(out,(q,k,v),grad,retain_graph=True)
    b=torch.autograd.grad(out,(q,k,v),grad)
    # Native efficient backward may choose nondeterministic accumulation order.
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-3,atol=2e-4)


@CUDA
def test_invalid_lengths_and_unequal_heads_rejected():
    from solution.attention_rows import make_layout,per_row_causal
    for lengths in ([0,2],[3,5],[]):
        with pytest.raises(ValueError):make_layout(lengths,4,'cuda')
    q=torch.zeros((2,12,4,128),device='cuda',dtype=torch.float16)
    k=torch.zeros((2,2,4,128),device='cuda',dtype=torch.float16)
    with pytest.raises(ValueError):per_row_causal(q,k,k,make_layout([4,3],4,'cuda'))

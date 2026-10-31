"""The measured large-shape dispatch keeps the dense path for small batches."""
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires Kaggle T4 CUDA')


@pytest.mark.parametrize('sequence',[128,384,512])
def test_batch_dropout_dispatch_preserves_values_gradients_and_rng(monkeypatch,sequence):
    from solution import unpad
    batch,width=4,1536
    lengths=[sequence,max(1,sequence//2),max(1,sequence//3),max(1,sequence//5)]
    mask=torch.arange(sequence,device='cuda')[None]<torch.tensor(lengths,device='cuda')[:,None]
    rows=mask.reshape(-1).nonzero().squeeze(1)
    torch.manual_seed(4711)
    x=torch.randn((rows.numel(),width),device='cuda',requires_grad=True)
    reference=x.detach().clone().requires_grad_(True)
    qgrad,vgrad=torch.randn_like(x),torch.randn_like(x)
    module=torch.nn.Dropout(.05).train()
    def dense(value):
        padded=value.new_zeros((batch*sequence,width)).index_copy(0,rows,value).view(batch,sequence,width)
        q=module(padded).view(batch*sequence,width).index_select(0,rows)
        v=module(padded).view(batch*sequence,width).index_select(0,rows)
        return q,v
    state=torch.cuda.get_rng_state()
    expected=dense(reference)
    expected_state=torch.cuda.get_rng_state()
    expected_dx,=torch.autograd.grad(expected,reference,(qgrad,vgrad))
    torch.cuda.set_rng_state(state)
    calls=[];real=unpad.real_row_dropout
    def record(*args,**kwargs):
        calls.append(True);return real(*args,**kwargs)
    monkeypatch.setattr(unpad,'real_row_dropout',record)
    step=unpad._Batch(rows,batch,sequence)
    actual=(step.dropout(module,x),step.dropout(module,x))
    step.done_with(x)
    actual_state=torch.cuda.get_rng_state()
    actual_dx,=torch.autograd.grad(actual,x,(qgrad,vgrad))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(actual_dx,expected_dx,rtol=0,atol=0)
    assert torch.equal(actual_state,expected_state)
    assert len(calls)==(2 if sequence>=384 else 0)

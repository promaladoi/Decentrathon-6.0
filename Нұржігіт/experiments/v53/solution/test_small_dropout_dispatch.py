"""Qualify the actual small-batch dispatcher against dense native dropout."""
import pytest
import torch
from torch.nn import functional as F

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA/Triton')


@pytest.mark.parametrize('batch,sequence,width,lengths',[
    (4,64,1536,(64,51,19,7)),(4,127,1536,(127,100,13,1)),
    (4,256,1536,(256,192,32,9)),(4,383,1536,(383,351,299,190)),
    (2,17,31,(17,9))])
def test_small_dispatch_output_gradient_rng_and_packed_storage_are_exact(batch,sequence,width,lengths):
    from solution.unpad import _Batch
    torch.manual_seed(5301)
    mask=torch.arange(sequence,device='cuda')[None] < torch.tensor(lengths,device='cuda')[:,None]
    rows=mask.reshape(-1).nonzero().squeeze(1)
    x=torch.randn(rows.numel(),width,device='cuda',requires_grad=True)
    upstream=torch.randn_like(x)
    state=torch.cuda.get_rng_state()
    padded=x.new_zeros(batch*sequence,width).index_copy(0,rows,x).view(batch,sequence,width)
    expected=F.dropout(padded,p=.05,training=True).view(-1,width).index_select(0,rows)
    after=torch.cuda.get_rng_state()
    next_draw=torch.rand(19,device='cuda')
    expected_grad,=torch.autograd.grad(expected,x,upstream)
    torch.cuda.set_rng_state(state)
    saved=[]
    step=_Batch(rows,batch,sequence)
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        actual=step.dropout(torch.nn.Dropout(.05),x)
    assert not step._padded
    assert len(saved)==1 and saved[0].dtype==torch.uint8
    assert saved[0].numel()==(x.numel()+7)//8
    assert torch.equal(torch.cuda.get_rng_state(),after)
    assert torch.equal(torch.rand(19,device='cuda'),next_draw)
    actual_grad,=torch.autograd.grad(actual,x,upstream)
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert torch.equal(actual_grad.view(torch.int32),expected_grad.view(torch.int32))


def test_small_module_hook_still_observes_native_padded_call():
    from solution.unpad import _Batch
    rows=torch.tensor([0,1,2,5,6],device='cuda')
    x=torch.randn(5,31,device='cuda',requires_grad=True)
    module=torch.nn.Dropout(.05)
    observed=[]
    handle=module.register_forward_hook(lambda layer,args,output:observed.append(tuple(args[0].shape)))
    try:
        result=_Batch(rows,2,5).dropout(module,x)
        assert result.shape==x.shape and observed==[(2,5,31)]
    finally:
        handle.remove()

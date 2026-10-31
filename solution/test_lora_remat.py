"""Qualify rematerialized LoRA inputs against the unchanged v45 arithmetic."""
import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _rows(batch, sequence, lengths):
    valid = torch.arange(sequence,device='cuda')[None] < torch.tensor(lengths,device='cuda')[:,None]
    return valid.reshape(-1).nonzero().squeeze(1)


def _exact(actual,expected):
    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    integer=torch.int32 if actual.dtype==torch.float32 else torch.int16
    assert torch.equal(actual.contiguous().view(integer),expected.contiguous().view(integer))


@pytest.mark.parametrize('width,sequence,lengths',[(31,17,(17,9)),(1536,384,(384,211,37,90))])
@pytest.mark.parametrize('p',[0.05,0.37])
def test_two_lora_branches_exact_values_gradients_rng_and_retained_graph(width,sequence,lengths,p):
    from solution.dropout import real_row_dropout
    from solution.rmsnorm import add_rmsnorm,add_rmsnorm_with_stats
    from solution.lora_remat import lora_input_projection
    torch.manual_seed(640)
    batch=len(lengths)
    rows=_rows(batch,sequence,lengths)
    shape=(rows.numel(),width)
    residual=torch.randn(shape,device='cuda',requires_grad=True)
    delta=torch.randn(shape,device='cuda',dtype=torch.float16,requires_grad=True)
    norm_weight=torch.randn((width,),device='cuda')*0.1+1.0
    a=[(torch.randn((16,width),device='cuda')*0.01).requires_grad_() for _ in range(2)]
    r2=residual.detach().clone().requires_grad_()
    d2=delta.detach().clone().requires_grad_()
    a2=[value.detach().clone().requires_grad_() for value in a]
    grad=[torch.randn((shape[0],16),device='cuda',dtype=torch.float16)*32 for _ in range(2)]
    initial=torch.cuda.get_rng_state()
    _,x=add_rmsnorm(residual,delta,norm_weight,1e-6)
    with torch.autocast('cuda',dtype=torch.float16):
        reference=[torch.nn.functional.linear(real_row_dropout(
            x,rows,batch,sequence,p,rows_from_nonzero=True),value) for value in a]
    after=torch.cuda.get_rng_state()
    next_reference=torch.rand(19,device='cuda')
    expected=torch.autograd.grad(reference,(residual,delta,*a),grad,retain_graph=True)
    torch.cuda.set_rng_state(initial)
    h,x,rstd=add_rmsnorm_with_stats(r2,d2,norm_weight,1e-6)
    assert not rstd.requires_grad
    saved=[]
    with torch.autocast('cuda',dtype=torch.float16),torch.autograd.graph.saved_tensors_hooks(
            lambda t:saved.append(t) or t,lambda t:t):
        actual=[lora_input_projection(x,value,(h,norm_weight,rstd),rows,batch,sequence,p) for value in a2]
    assert torch.equal(torch.cuda.get_rng_state(),after)
    assert torch.equal(torch.rand(19,device='cuda'),next_reference)
    # Two branches share the already-required RMS residual; neither retains
    # a [real_tokens,hidden] FP16 or FP32 normalized/dropout input allocation.
    assert len(saved)==10
    assert all(t.dtype!=torch.float16 or t.shape==(16,width) for t in saved)
    for branch in (saved[:5],saved[5:]):
        assert branch[1].data_ptr()==h.data_ptr()
        assert branch[-1].dtype==torch.uint8 and branch[-1].numel()==(x.numel()+7)//8
    before_backward=torch.cuda.get_rng_state()
    observed=torch.autograd.grad(actual,(r2,d2,*a2),grad,retain_graph=True)
    assert torch.equal(torch.cuda.get_rng_state(),before_backward)
    for observed_output,expected_output in zip(actual,reference): _exact(observed_output,expected_output)
    for value,target in zip(observed,expected): _exact(value,target)
    # Retained graphs must permit a second backward without modifying the
    # saved residual, inverse RMS, dropout bits or half LoRA weight.
    again=torch.autograd.grad(actual,(r2,d2,*a2),grad)
    for value,target in zip(again,expected): _exact(value,target)


@pytest.mark.parametrize('hooked',['none','linear','dropout','override','frozen','no_amp'])
def test_dispatch_preserves_observers_and_native_fallback(monkeypatch,hooked):
    from solution import lora_remat
    from solution.rmsnorm import add_rmsnorm_with_stats
    from solution.unpad import _Batch,_lora_out
    torch.manual_seed(641)
    rows=_rows(4,384,(384,201,27,110))
    residual=torch.randn((rows.numel(),1536),device='cuda',requires_grad=True)
    h,x,rstd=add_rmsnorm_with_stats(residual,torch.zeros_like(residual,dtype=torch.float16),
                                  torch.ones(1536,device='cuda'),1e-6)
    norm_weight=torch.ones(1536,device='cuda')
    a=nn.Linear(1536,16,bias=False,device='cuda',dtype=torch.float32)
    b=nn.Linear(16,256,bias=False,device='cuda',dtype=torch.float32)
    dropout=nn.Dropout(0.05)
    calls=[]
    if hooked=='linear': a.register_forward_hook(lambda *args:calls.append('observer'))
    if hooked=='dropout': dropout.register_forward_hook(lambda *args:calls.append('observer'))
    if hooked=='override':
        native_linear_forward=a.forward
        def replaced(value):
            calls.append('observer')
            return native_linear_forward(value)
        a.forward=replaced
    if hooked=='frozen': a.weight.requires_grad_(False)
    remat_projection=lora_remat.lora_input_projection
    def traced(*args):
        calls.append('remat')
        return remat_projection(*args)
    monkeypatch.setattr(lora_remat,'lora_input_projection',traced)
    step=_Batch(rows,4,384,packed_metadata={})
    with torch.autocast('cuda',dtype=torch.float16,enabled=hooked!='no_amp'):
        result=_lora_out((None,a,b,dropout,2.0),x,step,norm_descriptor=(h,norm_weight,rstd))
    assert result.shape==(rows.numel(),256)
    assert ('remat' in calls)==(hooked=='none')
    if hooked in ('linear','dropout','override'): assert 'observer' in calls


def test_exposed_rms_statistics_keep_original_output_and_backward():
    from solution.rmsnorm import add_rmsnorm,add_rmsnorm_with_stats
    torch.manual_seed(642)
    residual=torch.randn((37,1536),device='cuda',requires_grad=True)
    delta=torch.randn_like(residual,dtype=torch.float16,requires_grad=True)
    weight=torch.randn(1536,device='cuda',requires_grad=True)
    other=[value.detach().clone().requires_grad_() for value in (residual,delta,weight)]
    expected=add_rmsnorm(residual,delta,weight,1e-6)
    actual=add_rmsnorm_with_stats(*other,1e-6)
    grads=[torch.randn_like(value) for value in expected]
    expected_grad=torch.autograd.grad(expected,(residual,delta,weight),grads)
    actual_grad=torch.autograd.grad(actual[:2],other,grads)
    for value,target in zip(actual[:2],expected): _exact(value,target)
    for value,target in zip(actual_grad,expected_grad): _exact(value,target)

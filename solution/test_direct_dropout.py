"""Exact production qualification, including every native RNG bit and progression."""
import importlib.util
from pathlib import Path
import pytest
import torch

from solution import direct_dropout as direct


@pytest.mark.parametrize('seed',[1,690,2**48+17])
@pytest.mark.parametrize('p',[0.05,0.5,0.9])
@pytest.mark.parametrize('batch,sequence,width',[(4,512,1536),(4,384,1536),(1,13,256),(2,8,8)])
def test_every_real_mask_bit_value_and_native_counter_progression(seed,p,batch,sequence,width):
    xfull=torch.randn((batch,sequence,width),device='cuda')
    rows=torch.arange(batch*sequence,device='cuda')[::3].contiguous()
    x=xfull.reshape(-1,width).index_select(0,rows)
    generator=torch.cuda.default_generators[x.device.index]
    torch.cuda.manual_seed(seed)
    for _ in range(3):
        offset=generator.get_offset()
        reference,mask=torch.ops.aten.native_dropout.default(xfull,p,True)
        after=generator.get_offset()
        actual,packed,increment=direct.gather(x,rows,batch,sequence,p,seed,offset)
        expected=reference.reshape(-1,width).index_select(0,rows)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        bits=((packed[:,None].to(torch.int32)>>torch.arange(8,device='cuda'))&1).reshape(-1)[:x.numel()]
        torch.testing.assert_close(bits.bool(),mask.reshape(-1,width).index_select(0,rows).reshape(-1))
        assert generator.get_offset()==after==offset+increment
    # A private reservation that advances by the proven native count must
    # preserve the exact following random draw, not only the current mask.
    torch.cuda.manual_seed(seed)
    for _ in range(3): torch.ops.aten.native_dropout.default(xfull,p,True)
    following=torch.rand(1000,device='cuda')
    torch.cuda.manual_seed(seed)
    for _ in range(3):
        offset=generator.get_offset()
        _,_,increment=direct.gather(x,rows,batch,sequence,p,seed,offset)
        generator.set_offset(offset+increment)
    torch.testing.assert_close(torch.rand(1000,device='cuda'),following,rtol=0,atol=0)


def test_exceptional_values_half_output_and_backward_mask_bits():
    from solution.dropout import real_dropout_backward_kernel,_scales,BLOCK
    import triton
    batch,sequence,width=4,512,1536
    row=torch.arange(batch*sequence,device='cuda')[::2].contiguous()
    x=torch.randn((row.numel(),width),device='cuda')
    x.reshape(-1)[:8]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),1.,-1.,2.],device='cuda')
    full=torch.zeros((batch*sequence,width),device='cuda').index_copy(0,row,x).view(batch,sequence,width)
    generator=torch.cuda.default_generators[0]
    torch.cuda.manual_seed(701)
    offset=generator.get_offset()
    ref,mask=torch.ops.aten.native_dropout.default(full,.05,True)
    output,packed,_=direct.gather(x,row,batch,sequence,.05,701,offset,torch.float16)
    expected=ref.view(-1,width).index_select(0,row).half()
    assert torch.equal(output.view(torch.int16),expected.view(torch.int16))
    gradient=torch.randn_like(x)
    dx=torch.empty_like(x)
    n=x.numel()
    scale=_scales(.05)[1]
    real_dropout_backward_kernel[(triton.cdiv(n,BLOCK),)](
        gradient,packed,dx,n,scale,BLOCK,num_warps=4,enable_fp_fusion=False)
    native_mask=mask.view(-1,width).index_select(0,row)
    expected_dx=torch.ops.aten.native_dropout_backward.default(gradient,native_mask,1/(1-.05))
    torch.testing.assert_close(dx,expected_dx,rtol=0,atol=0)


@pytest.mark.parametrize('p',[.05,.37])
@pytest.mark.parametrize('scope_loss_scale',[None,65536.])
def test_private_lora_projection_uses_direct_mask_and_preserves_all_gradients(monkeypatch,p,scope_loss_scale):
    from solution import private_scope
    from solution.test_lora_remat import test_two_lora_branches_exact_values_gradients_rng_and_retained_graph
    original=direct.try_gather
    used=[]
    def traced(*args):
        result=original(*args)
        if result is not None: used.append(True)
        return result
    monkeypatch.setattr(direct,'try_gather',traced)
    token=private_scope._SCOPE.set(private_scope._Scope(None,None,None,None,scope_loss_scale))
    try:
        test_two_lora_branches_exact_values_gradients_rng_and_retained_graph(
            1536,384,(384,211,37,90),p)
    finally:
        private_scope._SCOPE.reset(token)
    assert len(used)==2


@pytest.mark.parametrize('unsupported',['public','worker','capture'])
def test_unsupported_consumer_does_not_reserve_or_draw_rng(monkeypatch,unsupported):
    from solution import private_scope
    x=torch.randn((10,1536),device='cuda')
    rows=torch.arange(10,device='cuda')
    token=private_scope._SCOPE.set(None if unsupported=='public' else
                                  private_scope._Scope(None,None,None,None,65536.))
    if unsupported=='worker': monkeypatch.setattr(direct,'_known_rng_threads',lambda:False)
    if unsupported=='capture': monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    initial=torch.cuda.get_rng_state()
    try: assert direct.try_gather(x,rows,4,512,.05) is None
    finally: private_scope._SCOPE.reset(token)
    assert torch.equal(torch.cuda.get_rng_state(),initial)

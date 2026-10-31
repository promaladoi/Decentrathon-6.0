"""Inactive numerical screen, including every native RNG bit and progression."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('direct_dropout',Path(__file__).with_name('direct_dropout.py'))
direct=importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)


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

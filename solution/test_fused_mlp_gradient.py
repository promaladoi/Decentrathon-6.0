import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='T4 CUDA required')


@pytest.mark.parametrize('shape',[(1,37,19),(23,128,256),(127,384,1024),(833,1536,8960)])
@pytest.mark.parametrize('config',[(32,64,32,4),(32,64,64,4),(64,64,64,4),(32,128,64,4)])
def test_independent_half_rounding_and_fp32_sum(shape,config):
    from .fused_mlp_gradient import fused_gradient
    m,n,k=shape
    torch.manual_seed(17)
    dg,du=[torch.randn((m,k),device='cuda',dtype=torch.float16)*.1 for _ in range(2)]
    wg,wu=[torch.randn((k,n),device='cuda',dtype=torch.float16)*.1 for _ in range(2)]
    partial_g,partial_u=dg@wg,du@wu
    expected=partial_g.float()+partial_u.float()
    actual=fused_gradient(dg,du,wg,wu,config)
    assert actual.dtype==torch.float32 and actual.shape==(m,n)
    relative=torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)
    assert relative.item()<.001
    # CUBLAS and Triton can accumulate the dot in different FP32 orders before
    # each FP16 output rounding. A fixed tolerance on their final sum fails
    # near cancellation: two of 1.28M entries differed by 0.00390625 even while
    # the strict global L2 bound passed. Bound per-entry error by the ulps of
    # the independent partial products, not by the small cancelled sum.
    def ulp(x):
        p=torch.nextafter(x,torch.full_like(x,float('inf'))).float()-x.float()
        q=x.float()-torch.nextafter(x,torch.full_like(x,-float('inf'))).float()
        return torch.maximum(p.abs(),q.abs())
    error=(actual-expected).abs()
    bound=2*(ulp(partial_g)+ulp(partial_u))+1e-5
    assert torch.all(error<=bound).item()


def test_round_each_partial_before_sum_instead_of_rounding_combined_dot():
    from .fused_mlp_gradient import fused_gradient
    g=torch.tensor([[1.0,0.0003]],device='cuda',dtype=torch.float16)
    u=torch.tensor([[-1.0,0.0003]],device='cuda',dtype=torch.float16)
    w=torch.ones((2,1),device='cuda',dtype=torch.float16)
    expected=(g@w).float()+(u@w).float()
    wrong=((g.float()@w.float())+(u.float()@w.float())).half().float()
    assert not torch.equal(expected,wrong)
    torch.testing.assert_close(fused_gradient(g,u,w,w),expected,atol=0,rtol=0)

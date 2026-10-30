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
    expected=(dg@wg).float()+(du@wu).float()
    actual=fused_gradient(dg,du,wg,wu,config)
    assert actual.dtype==torch.float32 and actual.shape==(m,n)
    relative=torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)
    assert relative.item()<.001
    torch.testing.assert_close(actual,expected,atol=.003,rtol=.008)

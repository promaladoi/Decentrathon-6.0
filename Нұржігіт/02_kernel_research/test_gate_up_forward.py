"""Independent value/backward-compatibility checks; inactive production path."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='T4 CUDA required')


def relative(actual, expected):
    return ((actual.float()-expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize('shape', [(1, 19, 37), (17, 257, 131), (833, 8960, 1536)])
@pytest.mark.parametrize('config', [(32, 64, 32, 4), (32, 128, 32, 4),
                                   (64, 64, 32, 4), (64, 128, 32, 4)])
def test_projection_rounding_activation_and_backward_compatibility(shape, config):
    from gate_up_forward import fused_gate_up, reference_gate_up
    m, n, k = shape
    torch.manual_seed(42)
    x = torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
    wg,wu = [torch.randn(n,k,device='cuda',dtype=torch.float16)*.03 for _ in range(2)]
    expected = reference_gate_up(x,wg,wu)
    actual = fused_gate_up(x,wg,wu,config)
    for a,b in zip(actual,expected):
        assert relative(a,b) < .001
        torch.testing.assert_close(a,b,rtol=.003,atol=1e-4)
    # The production backward would still use the original elementwise formula
    # and native GEMMs. Its input gradient must agree with those original values.
    dh = torch.randn_like(actual[2])*.1
    def backward(parts):
        g,u,_ = parts
        g,u=g.float(),u.float()
        sig=1/(1+torch.exp(-g))
        dg=(dh.float()*u*sig*(1+g*(1-sig))).half()
        du=(dh.float()*g*sig).half()
        return (dg@wg).float()+(du@wu).float()
    assert relative(backward(actual),backward(expected)) < .001


def test_zero_input_and_asymmetric_up_weight_keep_exact_zero():
    from gate_up_forward import fused_gate_up
    x = torch.zeros(35,128,device='cuda',dtype=torch.float16)
    wg = torch.ones(257,128,device='cuda',dtype=torch.float16)
    wu = -wg
    for out in fused_gate_up(x,wg,wu):
        assert torch.count_nonzero(out).item() == 0


def test_native_reference_agrees_with_the_actual_organizer_activation():
    from gate_up_forward import production_gate_up, reference_gate_up
    torch.manual_seed(43)
    x = torch.randn(37,128,device='cuda',dtype=torch.float16)*.1
    wg,wu = [torch.randn(257,128,device='cuda',dtype=torch.float16)*.03 for _ in range(2)]
    for actual,expected in zip(production_gate_up(x,wg,wu),reference_gate_up(x,wg,wu)):
        torch.testing.assert_close(actual,expected,rtol=.003,atol=1e-4)


def test_research_rejects_unsupported_layout_before_launch():
    from gate_up_forward import fused_gate_up
    x = torch.ones(8,16,device='cuda',dtype=torch.float16)[:,::2]
    w = torch.ones(32,8,device='cuda',dtype=torch.float16)
    with pytest.raises(ValueError,match='contiguous'):
        fused_gate_up(x,w,w)

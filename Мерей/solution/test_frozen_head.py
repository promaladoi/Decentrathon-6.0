import copy
import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA fp16 AMP')


def test_frozen_head_cpu_noop():
    from solution.frozen_head import install
    assert install(torch.nn.Linear(4, 8)) == 0


def test_frozen_head_rejects_modified_native_class_and_functional_forward(monkeypatch):
    import torch.nn.functional as functional
    from solution import frozen_head
    assert frozen_head._canonical_linear()
    original = torch.nn.Linear.forward
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Linear, 'forward', lambda self, x: original(self, x))
        assert not frozen_head._canonical_linear()
    original_functional = functional.linear
    with monkeypatch.context() as patch:
        patch.setattr(functional, 'linear', lambda *args: original_functional(*args))
        assert not frozen_head._canonical_linear()


def test_frozen_head_rejects_observers_including_global_backward_pre_hooks():
    from solution.frozen_head import _unobserved
    layer = torch.nn.Linear(4, 8, bias=False)
    assert _unobserved(layer)
    handle = layer.register_forward_hook(lambda module, args, result: result)
    try:
        assert not _unobserved(layer)
    finally:
        handle.remove()
    assert _unobserved(layer)
    handle = torch.nn.modules.module.register_module_full_backward_pre_hook(
        lambda module, grad_output: None)
    try:
        assert not _unobserved(layer)
    finally:
        handle.remove()
    assert _unobserved(layer)


def test_frozen_head_cpu_model_is_noop_even_with_cuda_available(monkeypatch):
    from solution import frozen_head
    layer = torch.nn.Linear(4, 8, bias=False).requires_grad_(False)
    original = layer.forward
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert frozen_head.install(layer) == 0
    assert layer.forward == original


@CUDA
@pytest.mark.parametrize('shape',[(7,64),(2,9,64)])
@pytest.mark.parametrize('scale',[1.,256.,32768.])
def test_frozen_head_native_amp_gradient_parity(shape,scale):
    from solution.frozen_head import _FrozenHead,_supported
    torch.manual_seed(317)
    layer = torch.nn.Linear(64,97,bias=False,device='cuda').requires_grad_(False)
    x=torch.randn(shape,device='cuda',requires_grad=True)
    ref=x.detach().clone().requires_grad_(True)
    with torch.autocast('cuda',dtype=torch.float16):
        assert _supported(layer,x)
        actual=_FrozenHead.apply(x,layer.weight)
        expected=layer(ref)
    assert actual.dtype == expected.dtype == torch.float16
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream=torch.randn_like(actual)*scale
    a=torch.autograd.grad(actual,x,upstream)[0]
    b=torch.autograd.grad(expected,ref,upstream)[0]
    assert a.dtype == b.dtype == torch.float32
    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


@CUDA
def test_frozen_head_real_qwen_install_and_saved_weight():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from solution import frozen_head
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=97,hidden_size=64,
        intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,
        num_key_value_heads=2,tie_word_embeddings=False)).cuda().requires_grad_(False)
    original=copy.deepcopy(model.lm_head)
    assert frozen_head.install(model)==1
    assert frozen_head.install(model)==0
    x=torch.randn((2,9,64),device='cuda',requires_grad=True)
    ref=x.detach().clone().requires_grad_(True)
    saved=[]
    def pack(t): saved.append((t.shape,t.dtype));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        with torch.autocast('cuda',dtype=torch.float16):actual=model.lm_head(x)
    with torch.autocast('cuda',dtype=torch.float16):expected=original(ref)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert (torch.Size([97,64]),torch.float32) in saved
    assert not any(shape==torch.Size([97,64]) and dtype==torch.float16 for shape,dtype in saved)
    dy=torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual,x,dy)[0],
                               torch.autograd.grad(expected,ref,dy)[0],rtol=0,atol=0)

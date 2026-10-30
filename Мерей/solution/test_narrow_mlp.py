"""Exact qualification of cast movement through actual frozen bnb MLPs."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _quantized_mlp():
    bnb = pytest.importorskip("bitsandbytes")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    config = Qwen2Config(hidden_size=64, intermediate_size=192,
                         num_hidden_layers=1, num_attention_heads=1,
                         num_key_value_heads=1, hidden_act="silu")
    mlp = Qwen2MLP(config)
    for name in ("gate_proj", "up_proj", "down_proj"):
        before = getattr(mlp, name)
        layer = bnb.nn.Linear4bit(before.in_features, before.out_features,
                                 bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4")
        layer.weight.data.copy_(before.weight)
        setattr(mlp, name, layer.to(torch.float16).to("cuda"))
    return mlp


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("scale", [1.0, 256.0, 65536.0])
def test_narrow_mlp_preserves_real_quantized_forward_and_input_gradient(
        monkeypatch, autocast, scale):
    pytest.importorskip("triton")
    from solution import narrow_mlp
    from solution.swiglu import swiglu
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(201)
    mlp = _quantized_mlp()
    x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x) * scale
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        reference = mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))
    reference_grad, = torch.autograd.grad(reference, (x,), grad)
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    narrow_mlp.patch_qwen2_mlp()
    installed = Qwen2MLP.forward
    narrow_mlp.patch_qwen2_mlp()
    assert installed is Qwen2MLP.forward
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = mlp(x)
    actual_grad, = torch.autograd.grad(actual, (x,), grad)
    assert actual.dtype == reference.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0,
                               equal_nan=True)


def test_narrow_mlp_saves_half_activations_and_keeps_fp32_input_branch_sum(monkeypatch):
    pytest.importorskip("triton")
    from solution import narrow_mlp
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(202)
    mlp = _quantized_mlp()
    x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    narrow_mlp.patch_qwen2_mlp()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append((tensor.shape, tensor.dtype)) or tensor,
            lambda tensor: tensor):
        actual = mlp(x)
    activations = [dtype for shape, dtype in saved if shape == (2, 31, 192)]
    assert activations == [torch.float16, torch.float16]
    gradient, = torch.autograd.grad(actual, (x,), torch.ones_like(actual))
    assert gradient.dtype == torch.float32

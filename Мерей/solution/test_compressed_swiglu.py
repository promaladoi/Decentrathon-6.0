"""GPU-only qualification; these tests do not alter the task or harness."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("shape", [(33, 192), (4, 127, 8960)])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_saved_half_storage_preserves_example_outputs_and_fp32_gradients(shape, scale):
    pytest.importorskip("triton")
    from solution.compressed_swiglu import CompressedSwiGLU
    from solution.swiglu import SwiGLU

    torch.manual_seed(199)
    gate = torch.randn(shape, device="cuda", dtype=torch.float16).float().requires_grad_()
    up = torch.randn(shape, device="cuda", dtype=torch.float16).float().requires_grad_()
    grad = torch.randn(shape, device="cuda", dtype=torch.float32) * scale
    reference = SwiGLU.apply(gate, up)
    reference_grad = torch.autograd.grad(reference, (gate, up), grad)
    packed = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: packed.append((tensor.shape, tensor.dtype)) or tensor,
            lambda tensor: tensor):
        actual = CompressedSwiGLU.apply(gate, up)
    actual_grad = torch.autograd.grad(actual, (gate, up), grad)
    assert packed == [(gate.shape, torch.float16), (up.shape, torch.float16)]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual_value, expected_value in zip(actual_grad, reference_grad):
        assert actual_value.dtype == expected_value.dtype == torch.float32
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


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
        layer = layer.to(torch.float16).to("cuda")
        setattr(mlp, name, layer)
    return mlp


def test_real_plain_bnb_mlp_forward_and_input_gradient_are_bitexact(monkeypatch):
    pytest.importorskip("triton")
    from solution import compressed_swiglu
    from solution.swiglu import swiglu
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(200)
    mlp = _quantized_mlp()
    x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    assert compressed_swiglu._eligible(mlp, x)
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    torch.testing.assert_close(gate, gate.half().float(), rtol=0, atol=0)
    torch.testing.assert_close(up, up.half().float(), rtol=0, atol=0)
    reference = mlp.down_proj(swiglu(gate, up))
    grad = torch.randn_like(reference) * 256.0
    reference_grad, = torch.autograd.grad(reference, (x,), grad)
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    compressed_swiglu.patch_qwen2_mlp()
    installed = Qwen2MLP.forward
    compressed_swiglu.patch_qwen2_mlp()
    assert installed is Qwen2MLP.forward
    actual = mlp(x)
    actual_grad, = torch.autograd.grad(actual, (x,), grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_guard_rejects_trainable_and_adapter_wrapped_projections():
    pytest.importorskip("triton")
    from solution.compressed_swiglu import _eligible

    mlp = _quantized_mlp()
    x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32)
    assert _eligible(mlp, x)
    assert not _eligible(mlp, x.half())
    old = mlp.gate_proj
    mlp.gate_proj = torch.nn.Sequential(old)
    assert not _eligible(mlp, x)
    mlp.gate_proj = old
    mlp.gate_proj.compute_dtype = torch.float32
    assert not _eligible(mlp, x)


def test_guard_rejects_projection_hooks_and_replaced_forwards(monkeypatch):
    pytest.importorskip("triton")
    from solution.compressed_swiglu import _eligible

    mlp = _quantized_mlp()
    x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32)
    assert _eligible(mlp, x)
    handle = mlp.gate_proj.register_forward_hook(lambda module, args, result: result)
    assert not _eligible(mlp, x)
    handle.remove()
    assert _eligible(mlp, x)
    before = mlp.gate_proj.forward
    monkeypatch.setattr(mlp.gate_proj, "forward", lambda value: before(value))
    assert not _eligible(mlp, x)

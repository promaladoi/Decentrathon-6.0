"""Correctness of the fused residual-add RMSNorm, RoPE, the 4-bit fp16 MLP and the patched
decoder layer against the original transformers / bitsandbytes code, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _close(a, b, tol):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("delta_dtype,res_dtype", [(torch.float16, torch.float32),
                                                   (torch.float32, torch.float32),
                                                   (torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 300, 1536), (2, 33, 1000)])
def test_add_rmsnorm_matches_pytorch(delta_dtype, res_dtype, shape):
    from solution.team_vladimir.rmsnorm import add_rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    residual = torch.randn(shape, device="cuda", dtype=res_dtype, requires_grad=True)
    delta = torch.randn(shape, device="cuda", dtype=delta_dtype, requires_grad=True)
    weight = 1 + 0.1 * torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    g_h = torch.randn(shape, device="cuda", dtype=torch.float32)
    g_y = torch.randn(shape, device="cuda", dtype=torch.float32)

    # transformers: hidden = residual + attn_out; Qwen2RMSNorm(hidden); hidden feeds the next add
    h = residual + delta
    x = h.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * x.to(h.dtype)
    torch.autograd.backward((h, y), (g_h.to(h.dtype), g_y.to(y.dtype)))
    ref = (h.detach(), y.detach(), residual.grad.clone(), delta.grad.clone())
    residual.grad = delta.grad = None

    h2, y2 = add_rmsnorm(residual, delta, weight, eps)
    assert h2.dtype == h.dtype and y2.dtype == y.dtype
    torch.autograd.backward((h2, y2), (g_h.to(h2.dtype), g_y.to(y2.dtype)))
    tol = 2e-3 if torch.float16 in (res_dtype, delta_dtype) else 1e-4
    _close(h2, ref[0], tol)
    _close(y2, ref[1], tol)
    assert residual.grad.dtype == res_dtype and delta.grad.dtype == delta_dtype
    _close(residual.grad, ref[2], tol)
    _close(delta.grad, ref[3], tol)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(batch, seq_len, dim, dtype):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    pos = torch.arange(seq_len, device="cuda").float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[None].expand(batch, -1, -1)
    return emb.cos().to(dtype).contiguous(), emb.sin().to(dtype).contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("cos_batch", [1, 3])
def test_rope_matches_transformers(dtype, cos_batch):
    from solution.team_vladimir.rope import RoPE

    torch.manual_seed(0)
    batch, seq_len, hq, hk, dim = 3, 37, 12, 2, 128
    # as in Qwen2Attention: projections viewed as (B, L, H, D) and transposed to (B, H, L, D)
    q_src = torch.randn(batch, seq_len, hq, dim, device="cuda", dtype=dtype, requires_grad=True)
    k_src = torch.randn(batch, seq_len, hk, dim, device="cuda", dtype=dtype, requires_grad=True)
    cos, sin = _cos_sin(cos_batch, seq_len, dim, dtype)
    gq = torch.randn(batch, hq, seq_len, dim, device="cuda", dtype=dtype)
    gk = torch.randn(batch, hk, seq_len, dim, device="cuda", dtype=dtype)

    q, k = q_src.transpose(1, 2), k_src.transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    ref_q = (q * c) + (_rotate_half(q) * s)
    ref_k = (k * c) + (_rotate_half(k) * s)
    torch.autograd.backward((ref_q, ref_k), (gq, gk))
    ref_dq, ref_dk = q_src.grad.clone(), k_src.grad.clone()
    q_src.grad = k_src.grad = None

    out_q, out_k = RoPE.apply(q_src.transpose(1, 2), k_src.transpose(1, 2), cos, sin)
    torch.autograd.backward((out_q, out_k), (gq, gk))
    # the same products and sums in the same dtype, without FMA: equal to rounding noise or better
    tol = 1e-3 if dtype == torch.float16 else 1e-6
    _close(out_q, ref_q, tol)
    _close(out_k, ref_k, tol)
    _close(q_src.grad, ref_dq, tol)
    _close(k_src.grad, ref_dk, tol)


def _to_4bit(linear):
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(linear.in_features, linear.out_features,
                              bias=linear.bias is not None, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(linear.weight.data.detach().to(torch.float16).cpu(),
                                     requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")
    if linear.bias is not None:
        layer.bias = torch.nn.Parameter(linear.bias.data.detach().float().cpu(),
                                        requires_grad=False)
    return layer.cuda()


def test_mlp_4bit_matches_bitsandbytes():
    import bitsandbytes.functional as bnbf
    from solution.team_vladimir.mlp import MLP4bit

    torch.manual_seed(0)
    hidden, inter = 256, 512
    gate = _to_4bit(torch.nn.Linear(hidden, inter, bias=False))
    up = _to_4bit(torch.nn.Linear(hidden, inter, bias=False))
    down = _to_4bit(torch.nn.Linear(inter, hidden, bias=False))
    x = torch.randn(3, 41, hidden, device="cuda", requires_grad=True)
    grad = torch.randn(3, 41, hidden, device="cuda")

    def w(layer):
        return bnbf.dequantize_4bit(layer.weight, layer.weight.quant_state).to(torch.float16)

    # transformers + Linear4bit: x cast to fp16 per layer, outputs cast back to fp32
    g = torch.nn.functional.linear(x.to(torch.float16), w(gate)).float()
    u = torch.nn.functional.linear(x.to(torch.float16), w(up)).float()
    h = torch.nn.functional.silu(g) * u
    ref = torch.nn.functional.linear(h.to(torch.float16), w(down)).float()
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    out = MLP4bit.apply(x, gate, up, down, torch.float16)
    out.float().backward(grad)
    assert out.dtype == torch.float16
    _close(out, ref, 2e-3)
    _close(x.grad, ref_dx, 2e-3)


def _tiny_layer():
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    config = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         rms_norm_eps=1e-6)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layer = modeling_qwen2.Qwen2DecoderLayer(config, 0)
    attn, mlp = layer.self_attn, layer.mlp
    for owner, name in ((attn, "q_proj"), (attn, "k_proj"), (attn, "v_proj"), (attn, "o_proj"),
                        (mlp, "gate_proj"), (mlp, "up_proj"), (mlp, "down_proj")):
        setattr(owner, name, _to_4bit(getattr(owner, name)))
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        norm.weight.data = 1 + 0.1 * torch.randn_like(norm.weight.data)
    layer = layer.cuda().train()
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config).cuda()
    return config, layer, rotary


def test_decoder_layer_matches_original(monkeypatch):
    import bitsandbytes
    from transformers.models.qwen2 import modeling_qwen2

    from solution.team_vladimir.layer import make_decoder_layer_forward
    from solution.team_vladimir.mlp import make_mlp_forward
    from solution.team_vladimir.nf4 import make_matmul_4bit
    from solution.team_vladimir.rmsnorm import make_rmsnorm_forward
    from solution.team_vladimir.rope import make_apply_rotary_pos_emb

    # both runs use the same 4-bit matmul, so the comparison isolates the layer-level patches
    monkeypatch.setattr(bitsandbytes, "matmul_4bit", make_matmul_4bit(bitsandbytes.matmul_4bit))
    config, layer, rotary = _tiny_layer()
    hidden = torch.randn(2, 24, config.hidden_size, device="cuda")
    position_ids = torch.arange(24, device="cuda")[None]
    cos, sin = rotary(hidden, position_ids)
    grad = torch.randn_like(hidden)

    def run():
        x = hidden.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, attention_mask=None, position_ids=position_ids,
                        position_embeddings=(cos, sin))
        out.backward(grad)
        return out.detach(), x.grad

    ref_out, ref_dx = run()

    cls = modeling_qwen2
    monkeypatch.setattr(cls.Qwen2RMSNorm, "forward", make_rmsnorm_forward(cls.Qwen2RMSNorm.forward))
    monkeypatch.setattr(cls.Qwen2MLP, "forward", make_mlp_forward(cls.Qwen2MLP.forward))
    monkeypatch.setattr(cls.Qwen2DecoderLayer, "forward",
                        make_decoder_layer_forward(cls.Qwen2DecoderLayer.forward, cls.Qwen2RMSNorm))
    monkeypatch.setattr(cls, "apply_rotary_pos_emb",
                        make_apply_rotary_pos_emb(cls.apply_rotary_pos_emb))
    out, dx = run()
    assert out.dtype == ref_out.dtype == torch.float32
    _close(out, ref_out, 1e-2)
    _close(dx, ref_dx, 1e-2)


def test_warmup_signatures_compile():
    """kernel.warmup must accept the signatures warmup.py passes (the thread swallows errors)."""
    from solution import mlp, nf4, rmsnorm, rope

    for step in (nf4.warmup, rmsnorm.warmup, rope.warmup, mlp.warmup):
        step()

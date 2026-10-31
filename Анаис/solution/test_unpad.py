"""Correctness of the unpadded training forward (unpad.py) and the chunked lm_head + loss
(lmhead.py) against the original transformers / PEFT / TRL / Soup code, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


@pytest.mark.parametrize("with_num_items", [True, False])
def test_lm_head_loss_matches_cross_entropy(with_num_items):
    from trl.trainer.utils import entropy_from_logits

    from solution.lmhead import lm_head_loss

    torch.manual_seed(0)
    tokens, hidden, vocab = 700, 128, 10007  # several row chunks and vocabulary blocks
    x = torch.randn(tokens, hidden, device="cuda", requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda") * 0.05
    targets = torch.randint(0, vocab, (tokens,), device="cuda")
    targets[torch.rand(tokens, device="cuda") < 0.4] = -100
    num_items = (targets != -100).sum() if with_num_items else None

    # transformers: lm_head under autocast (fp16), ForCausalLMLoss in fp32
    logits = torch.nn.functional.linear(x.to(torch.float16), weight.to(torch.float16)).float()
    reduction = "sum" if with_num_items else "mean"
    ref = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100,
                                            reduction=reduction)
    if with_num_items:
        ref = ref / num_items
    (ref * 1024).backward()  # with a loss scale, as GradScaler does
    ref_dx = x.grad.clone()
    x.grad = None

    loss, stats = lm_head_loss(x, weight, targets, num_items)
    (loss * 1024).backward()
    assert abs(loss.item() - ref.item()) <= 1e-4 * abs(ref.item())
    assert _rel(x.grad, ref_dx) < 5e-3
    with torch.no_grad():
        ref_ent = entropy_from_logits(logits)
        ref_ce = torch.nn.functional.cross_entropy(logits, targets.clamp(min=0),
                                                   reduction="none")
    torch.testing.assert_close(stats["entropy"], ref_ent, rtol=1e-4, atol=1e-4)
    valid = targets != -100
    torch.testing.assert_close(stats["ce"][valid], ref_ce[valid], rtol=1e-4, atol=1e-4)
    assert (stats["ce"][~valid] == 0).all()
    assert (stats["argmax"] == logits.argmax(-1)).float().mean() > 0.999


def _padded_reference(q, lq, k, v, lv, cos, sin, rows, batch, seq_len, hq, hk, dim, rep):
    """transformers: projections (+ LoRA) in fp32, RoPE, repeat_kv, autocast cast to fp16, in the
    padded layout (zeros where the batch has no token)."""
    def rot(x, c, s):
        half = dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return (x * c) + (torch.cat((-x2, x1), dim=-1) * s)

    tokens = rows.numel()
    q32 = (q.float() + lq.float()).view(tokens, hq, dim)
    k32 = k.float().view(tokens, hk, dim)
    v32 = (v.float() + lv.float()).view(tokens, hk, dim)
    c, s = cos[:, None, :], sin[:, None, :]
    out = []
    for t32, heads in ((rot(q32, c, s), hq), (rot(k32, c, s).repeat_interleave(rep, 1), hk * rep),
                       (v32.repeat_interleave(rep, 1), hk * rep)):
        full = t32.new_zeros(batch * seq_len, heads, dim).index_copy(0, rows, t32)
        out.append(full.view(batch, seq_len, heads, dim).transpose(1, 2).to(torch.float16))
    return out


@pytest.mark.parametrize("repeat", [True, False])  # False: SDPA's enable_gqa (no padding)
def test_rope_pad_matches_reference(repeat):
    from solution.unpad import RoPEPad

    torch.manual_seed(0)
    batch, seq_len, hq, hk, dim = 3, 29, 12, 2, 128
    lengths = [29, 17, 5]
    rows = torch.cat([torch.arange(n) + b * seq_len for b, n in enumerate(lengths)]).cuda()
    tokens = rows.numel()

    def leaf(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)

    q, lq, k, v, lv = (leaf(tokens, hq * dim), leaf(tokens, hq * dim), leaf(tokens, hk * dim),
                       leaf(tokens, hk * dim), leaf(tokens, hk * dim))
    pos = (rows % seq_len).float()
    inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    rep = hq // hk if repeat else 1
    grads = [torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16)
             for heads in (hq, hk * rep, hk * rep)]

    ref = _padded_reference(q, lq, k, v, lv, cos, sin, rows, batch, seq_len, hq, hk, dim, rep)
    torch.autograd.backward(ref, grads)
    ref_grads = [t.grad.clone() for t in (q, lq, k, v, lv)]
    for t in (q, lq, k, v, lv):
        t.grad = None

    out = RoPEPad.apply(q, lq, k, None, v, lv, cos, sin, rows, batch, seq_len, hq, hk, dim, rep)
    torch.autograd.backward(out, grads)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-3)  # one fp16 ulp at most
    for a, b in zip((q, lq, k, v, lv), ref_grads):
        torch.testing.assert_close(a.grad, b, rtol=2e-3, atol=2e-3)


def test_unpad_heads_round_trip():
    from solution.unpad import UnpadHeads

    torch.manual_seed(0)
    batch, heads, seq_len, dim = 2, 12, 11, 128
    rows = torch.tensor([0, 1, 2, 3, 11, 12, 13], device="cuda")
    padded = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16,
                         requires_grad=True)
    flat = UnpadHeads.apply(padded, rows, seq_len)
    ref = padded.transpose(1, 2).reshape(batch * seq_len, heads * dim).index_select(0, rows)
    assert torch.equal(flat, ref)
    grad = torch.randn_like(flat)
    flat.backward(grad)
    expect = torch.zeros(batch * seq_len, heads * dim, device="cuda", dtype=torch.float16)
    expect.index_copy_(0, rows, grad)
    assert torch.equal(padded.grad, expect.view(batch, seq_len, heads, dim).transpose(1, 2))


def _tiny_peft_model():
    import peft
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.test_layer import _to_4bit

    config = Qwen2Config(vocab_size=3001, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = modeling_qwen2.Qwen2ForCausalLM(config)
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        for owner, name in ((attn, "q_proj"), (attn, "k_proj"), (attn, "v_proj"),
                            (attn, "o_proj"), (mlp, "gate_proj"), (mlp, "up_proj"),
                            (mlp, "down_proj")):
            setattr(owner, name, _to_4bit(getattr(owner, name)))
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.weight.data = 1 + 0.1 * torch.randn_like(norm.weight.data)
    model = model.cuda()
    model.is_loaded_in_4bit = True  # what from_pretrained sets; PEFT picks its 4-bit LoRA layer
    lora = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
                           task_type="CAUSAL_LM")
    model = peft.get_peft_model(model, lora)
    for name, param in model.named_parameters():
        if "lora_B" in name:  # non-zero, so the LoRA path and its dropout show in the loss
            param.data.normal_(0, 0.05)
    return model.train()


def _batch(lengths, seq_len, vocab):
    torch.manual_seed(1)
    batch = len(lengths)
    input_ids = torch.randint(0, vocab, (batch, seq_len), device="cuda")
    attention_mask = torch.zeros(batch, seq_len, dtype=torch.long, device="cuda")
    labels = input_ids.clone()
    for b, n in enumerate(lengths):
        attention_mask[b, :n] = 1
        labels[b, n:] = -100
        labels[b, : max(1, n // 3)] = -100  # the prompt: no loss
        input_ids[b, n:] = 0  # pad token
    return input_ids, attention_mask, labels


@pytest.mark.parametrize("lengths", [[23, 15, 6], [16, 16, 16]])
def test_fast_forward_matches_original(monkeypatch, lengths):
    import bitsandbytes
    from transformers.models.qwen2 import modeling_qwen2
    from trl.trainer.utils import entropy_from_logits

    from solution.lmhead import LazyLogits, make_row_losses
    from solution.nf4 import make_matmul_4bit
    from solution.unpad import make_causal_lm_forward

    # both runs use the same 4-bit matmul, so the comparison isolates the unpadded forward
    monkeypatch.setattr(bitsandbytes, "matmul_4bit", make_matmul_4bit(bitsandbytes.matmul_4bit))
    model = _tiny_peft_model()
    seq_len = max(lengths)
    input_ids, attention_mask, labels = _batch(lengths, seq_len, 3001)
    num_items = (labels[:, 1:] != -100).sum()
    lora = [p for n, p in model.named_parameters() if p.requires_grad]

    def run():
        for p in lora:
            p.grad = None
        torch.manual_seed(1234)  # the same dropout draws in both runs if the order is the same
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        use_cache=False, num_items_in_batch=num_items)
        (out.loss * 1024).backward()
        return out, [p.grad.clone() for p in lora]

    ref, ref_grads = run()
    cls = modeling_qwen2.Qwen2ForCausalLM
    monkeypatch.setattr(cls, "forward", make_causal_lm_forward(cls.forward))
    out, grads = run()

    assert isinstance(out.logits, LazyLogits) and torch.is_tensor(ref.logits)
    assert abs(out.loss.item() - ref.loss.item()) <= 2e-3 * abs(ref.loss.item())
    for a, b in zip(grads, ref_grads):
        assert _rel(a, b) < 2e-2
    logits = ref.logits.float()  # accelerate hands TRL fp32 logits
    mask = attention_mask.bool()
    with torch.no_grad():
        torch.testing.assert_close(out.logits.entropy()[mask], entropy_from_logits(logits)[mask],
                                   rtol=1e-3, atol=1e-3)
        pred = out.logits[..., :-1, :].contiguous().argmax(dim=-1)
        ref_pred = logits[..., :-1, :].contiguous().argmax(dim=-1)
        labelled = labels[:, 1:] != -100
        assert (pred == ref_pred)[labelled].float().mean() > 0.98
    try:
        from soup_cli.trainer import rewind_hf
    except ImportError:
        return
    losses, tokens = make_row_losses(rewind_hf.row_losses)(out.logits, labels)
    ref_losses, ref_tokens = rewind_hf.row_losses(logits, labels)
    assert tokens == ref_tokens
    assert all(abs(a - b) <= 2e-3 * max(abs(b), 1e-6) for a, b in zip(losses, ref_losses))


def test_warmup_signatures_compile_unpad():
    from solution import lmhead, unpad

    unpad.warmup()
    lmhead.warmup()

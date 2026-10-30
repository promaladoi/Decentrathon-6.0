import math

import pytest
import torch


def test_prevalidated_batch_plan_avoids_rebuilding_metadata():
    from types import SimpleNamespace
    from .varlen_attention import make_batch_init
    obj=SimpleNamespace()
    def original(self,rows,batch,seq): self.original=(rows,batch,seq)
    plan={'cu':'already validated','max_len':5}
    make_batch_init(original)(obj,'rows',2,5,plan=plan)
    assert obj._varlen_plan is plan and obj.original==('rows',2,5)


def test_dense_mask_elision_requires_all_causal_varlen_layers():
    from types import SimpleNamespace as S
    from .varlen_attention import can_skip_dense_mask
    layers=[(S(self_attn=S(is_causal=True)),None) for _ in range(3)]
    meta={'dim':128,'layers':layers}
    assert can_skip_dense_mask(S(_varlen_plan={'cu':'valid'}),meta)
    assert not can_skip_dense_mask(S(),meta)
    assert not can_skip_dense_mask(S(_varlen_plan=None),meta)
    assert not can_skip_dense_mask(S(_varlen_plan={}),dict(meta,dim=256))
    layers[-1][0].self_attn.is_causal=False
    assert not can_skip_dense_mask(S(_varlen_plan={}),meta)
from torch.nn import functional as F

from solution.varlen_attention import attention, make_attention, make_batch_init, make_plan


@pytest.mark.parametrize('positions,counts', [
    ([0, 1, 3, 8, 9, 10, 16], [3, 3, 1]),
    ([2, 3, 4, 5, 9, 11, 12, 13, 14, 18, 20], [4, 5, 2]),
])
def test_plan_preserves_logical_batch_boundaries(positions, counts):
    rows = torch.tensor(positions)
    plan = make_plan(rows, 3, 8)
    expected = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32)
    assert torch.equal(plan['cu'], expected)
    assert plan['max_len'] == max(counts)
    assert torch.equal(plan['rows'], torch.arange(len(positions)))


@pytest.mark.parametrize('rows', [torch.tensor([], dtype=torch.long), torch.tensor([0, 1, 8])])
def test_empty_sequence_keeps_original_attention(rows):
    assert make_plan(rows, 3, 8) is None


def reference(q, k, v, lengths):
    size = max(lengths)
    qs, ks, vs, start = [], [], [], 0
    for length in lengths:
        for source, dest in ((q, qs), (k, ks), (v, vs)):
            x = source[0, start:start+length].transpose(0, 1)
            dest.append(F.pad(x, (0, 0, 0, size-length)))
        start += length
    qp, kp, vp = torch.stack(qs), torch.stack(ks), torch.stack(vs)
    causal = torch.ones(size, size, dtype=torch.bool, device=q.device).tril()
    keys = torch.arange(size, device=q.device)[None, :] < torch.tensor(lengths, device=q.device)[:, None]
    mask = causal[None, None] & keys[:, None, None, :]
    out = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask, dropout_p=0.0,
                                         scale=1/math.sqrt(q.shape[-1]))
    return torch.cat([out[b, :, :length].transpose(0, 1) for b, length in enumerate(lengths)]).unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('lengths', [[23, 15, 6], [32, 32, 32], [17, 1, 5], [512, 384, 97, 12]])
@pytest.mark.parametrize('contiguous', [False, True])
def test_native_varlen_forward_gradients_and_rng(lengths, contiguous):
    n, heads, dim = sum(lengths), 12, 128
    def operand():
        x = torch.randn(1, n, heads, dim, device='cuda', dtype=torch.float16) if contiguous else \
            torch.randn(1, heads, n, dim, device='cuda', dtype=torch.float16).transpose(1, 2)
        return x.detach().requires_grad_()
    q, k, v = operand(), operand(), operand()
    seq = max(lengths)
    rows = torch.cat([torch.arange(length, device='cuda')+b*seq for b, length in enumerate(lengths)])
    plan = make_plan(rows, len(lengths), seq)
    expected = reference(q, k, v, lengths)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    rng = torch.cuda.get_rng_state()
    actual = attention(q, k, v, plan, 1/math.sqrt(dim))
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for a, b in zip(actual_grads, expected_grads):
        assert (a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-7) < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_sequences_and_future_tokens_cannot_leak_attention():
    lengths = [7, 5]
    rows = torch.tensor([*range(7), *range(8, 13)], device='cuda')
    plan = make_plan(rows, 2, 8)
    q, k, v = [torch.randn(1, 12, 4, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
    base = attention(q, k, v, plan, 1/8)
    changed = v.clone()
    changed[:, 7:] += 10
    other = attention(q, k, changed, plan, 1/8)
    assert torch.equal(base[:, :7], other[:, :7])
    changed = v.clone()
    changed[:, 6] += 10
    other = attention(q, k, changed, plan, 1/8)
    assert torch.equal(base[:, :6], other[:, :6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('lengths', [[23, 15, 6], [16, 16, 16]])
def test_model_loss_lora_gradients_metrics_and_dropout_rng(monkeypatch, lengths):
    import bitsandbytes
    from transformers.models.qwen2 import modeling_qwen2
    from solution.team_vladimir import unpad
    from solution.team_vladimir.nf4 import make_matmul_4bit
    from solution.team_vladimir.test_unpad import _tiny_peft_model, _batch, _rel
    monkeypatch.setattr(bitsandbytes, 'matmul_4bit', make_matmul_4bit(bitsandbytes.matmul_4bit))
    model = _tiny_peft_model()
    monkeypatch.setattr(modeling_qwen2.Qwen2ForCausalLM, 'forward',
                        unpad.make_causal_lm_forward(modeling_qwen2.Qwen2ForCausalLM.forward))
    inputs, mask, labels = _batch(lengths, max(lengths), 3001)
    lora = [p for p in model.parameters() if p.requires_grad]
    def run():
        for p in lora:
            p.grad = None
        torch.manual_seed(1234)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(input_ids=inputs, attention_mask=mask, labels=labels,
                        use_cache=False, num_items_in_batch=(labels[:, 1:] != -100).sum())
        (out.loss*1024).backward()
        return out, [p.grad.clone() for p in lora], torch.cuda.get_rng_state()
    expected, eg, rng = run()
    monkeypatch.setattr(unpad._Batch, '__init__', make_batch_init(unpad._Batch.__init__))
    monkeypatch.setattr(unpad, '_attention', make_attention(unpad._attention))
    actual, ag, after = run()
    assert torch.equal(rng, after)
    assert abs(actual.loss.item()-expected.loss.item()) <= 2e-3*abs(expected.loss.item())
    for a, b in zip(ag, eg):
        assert _rel(a, b) < 2e-2
    torch.testing.assert_close(actual.logits.entropy()[mask.bool()],
                               expected.logits.entropy()[mask.bool()], rtol=1e-3, atol=1e-3)

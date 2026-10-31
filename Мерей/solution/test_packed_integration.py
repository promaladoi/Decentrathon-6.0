"""Packed RoPE/native-varlen qualification; no weakened inherited assertions."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires Kaggle T4 CUDA')


def _data(lengths=(29,17,5),heads=12,kv_heads=2,dim=128,all_lora=False):
    torch.manual_seed(4701)
    sequence=max(lengths);batch=len(lengths)
    rows=torch.cat([torch.arange(n)+b*sequence for b,n in enumerate(lengths)]).cuda()
    tokens=rows.numel()
    def leaf(shape):
        return (torch.randn(shape,device='cuda',dtype=torch.float16)*.2).requires_grad_()
    q=leaf((tokens,heads*dim));lq=leaf(q.shape)
    k=leaf((tokens,kv_heads*dim));lk=leaf(k.shape) if all_lora else None
    v=leaf(k.shape);lv=leaf(v.shape)
    position=(rows%sequence).float()
    inv=1.0/(1e6**(torch.arange(0,dim,2,device='cuda').float()/dim))
    embedding=torch.cat([torch.outer(position,inv)]*2,dim=-1)
    cos,sin=embedding.cos(),embedding.sin()
    return (q,lq,k,lk,v,lv,cos,sin,rows,batch,sequence,heads,kv_heads,dim,heads//kv_heads)


def _gather_packed(padded,lengths):
    return torch.cat([padded[b,:,:n,:] for b,n in enumerate(lengths)],dim=1).unsqueeze(0)


@pytest.mark.parametrize('all_lora',[False,True])
@pytest.mark.parametrize('lengths',[(29,17,5),(16,16,16)])
def test_packed_rope_mapping_and_input_gradients_are_exact(lengths,all_lora):
    from solution.unpad import RoPEPad,RoPEPacked
    args=_data(lengths,all_lora=all_lora)
    leaves=[x for x in args[:6] if x is not None]
    padded=RoPEPad.apply(*args)
    packed_grad=[torch.randn_like(_gather_packed(out,lengths)) for out in padded]
    padded_grad=[]
    for output,gradient in zip(padded,packed_grad):
        full=torch.zeros_like(output);start=0
        for b,n in enumerate(lengths):
            full[b,:,:n,:]=gradient[0,:,start:start+n,:]
            start+=n
        padded_grad.append(full)
    torch.autograd.backward(padded,padded_grad)
    expected=[x.grad.clone() for x in leaves]
    for x in leaves:x.grad=None
    packed=RoPEPacked.apply(*args)
    for result,reference in zip(packed,padded):
        assert result.shape==(1,12,sum(lengths),128)
        torch.testing.assert_close(result,_gather_packed(reference,lengths),rtol=0,atol=0)
    torch.autograd.backward(packed,packed_grad)
    for x,reference in zip(leaves,expected):
        torch.testing.assert_close(x.grad,reference,rtol=0,atol=0)


@pytest.mark.parametrize('lengths',[(29,17,5),(16,16,16)])
def test_packed_native_attention_and_rope_gradients_match_padded_path(lengths):
    from solution.unpad import RoPEPad,RoPEPacked,UnpadHeads
    args=_data(lengths)
    leaves=[x for x in args[:6] if x is not None]
    rows,batch,sequence,heads,_,dim,_=args[8:]
    allowed=torch.arange(sequence,device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None]
    causal=torch.arange(sequence,device='cuda')[:,None]>=torch.arange(sequence,device='cuda')[None,:]
    mask=causal[None,None]&allowed[:,None,None,:]
    q,k,v=RoPEPad.apply(*args)
    dense=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,
        dropout_p=0.0,is_causal=False,scale=dim**-.5,enable_gqa=False)
    expected=UnpadHeads.apply(dense,rows,sequence)
    upstream=torch.randn_like(expected)
    reference_grad=torch.autograd.grad(expected,leaves,upstream)
    q,k,v=RoPEPacked.apply(*args)
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    result=torch.ops.aten._efficient_attention_forward.default(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),None,cu,cu,sequence,sequence,
        0.0,1,True,scale=dim**-.5)[0].reshape(rows.numel(),heads*dim)
    gradient=torch.autograd.grad(result,leaves,upstream)
    torch.testing.assert_close(result,expected,rtol=3e-3,atol=3e-4)
    for actual,reference in zip(gradient,reference_grad):
        relative=(actual.float()-reference.float()).norm()/reference.float().norm().clamp(min=1e-8)
        assert torch.isfinite(relative) and float(relative)<1e-2


@pytest.mark.parametrize('lengths',[[23,15,6],[16,16,16]])
def test_full_small_peft_loss_metrics_and_gradients_use_packed_path(monkeypatch,lengths):
    from solution import unpad
    from solution.test_unpad import test_fast_forward_matches_original
    calls=[]
    original=unpad.RoPEPacked.forward
    def record(ctx,*args):
        outputs=original(ctx,*args)
        calls.append(tuple(t.shape for t in outputs))
        return outputs
    monkeypatch.setattr(unpad.RoPEPacked,'forward',staticmethod(record))
    # Retain every inherited numerical assertion, including the 2% full-LoRA
    # gradient criterion, entropy, accuracy, row losses and exact token counts.
    test_fast_forward_matches_original(monkeypatch,lengths)
    assert calls
    assert all(shapes[0][0]==1 and shapes[0][2]==sum(lengths) for shapes in calls)


def test_batch_guards_reject_invalid_masks_or_removed_labels_without_rng_draw():
    from solution.unpad import _batch_metadata
    ids=torch.ones((2,5),device='cuda',dtype=torch.int64)
    mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],device='cuda')
    labels=ids.clone();labels[mask==0]=-100
    state=torch.cuda.get_rng_state().clone()
    metadata=_batch_metadata(ids,mask,labels)
    assert metadata is not None
    assert metadata['maximum']==5 and metadata['cumulative'].dtype==torch.int32
    assert metadata['cumulative'].tolist()==[0,5,8]
    assert torch.equal(state,torch.cuda.get_rng_state())
    bad_masks=[mask.clone(),mask.clone(),mask.clone(),mask.clone()]
    bad_masks[0][1,1]=2                # non-binary
    bad_masks[1][1]=torch.tensor([0,0,1,1,1],device='cuda')  # left padding
    bad_masks[2][1]=torch.tensor([1,0,1,1,0],device='cuda')  # a hole
    bad_masks[3][1].zero_()            # empty row
    for bad in bad_masks:assert _batch_metadata(ids,bad,labels) is None
    bad_labels=labels.clone();bad_labels[1,4]=7
    assert _batch_metadata(ids,mask,bad_labels) is None
    assert torch.equal(state,torch.cuda.get_rng_state())


def test_unsupported_batch_uses_original_before_consuming_dropout(monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2
    from solution.test_unpad import _tiny_peft_model,_batch
    from solution.unpad import make_causal_lm_forward
    model=_tiny_peft_model()
    ids,mask,labels=_batch([19,11],19,3001)
    # Unsupported left padding: comparisons concern the original public path.
    ids[1]=ids[1].roll(8);mask[1]=mask[1].roll(8);labels[1]=labels[1].roll(8)
    def call():
        with torch.autocast('cuda',dtype=torch.float16):
            return model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
    torch.manual_seed(4702);expected=call();state=torch.cuda.get_rng_state().clone()
    cls=modeling_qwen2.Qwen2ForCausalLM
    monkeypatch.setattr(cls,'forward',make_causal_lm_forward(cls.forward,private_only=False))
    torch.manual_seed(4702);actual=call()
    assert torch.is_tensor(actual.logits)
    torch.testing.assert_close(actual.logits,expected.logits,rtol=0,atol=0)
    torch.testing.assert_close(actual.loss,expected.loss,rtol=0,atol=0)
    assert torch.equal(state,torch.cuda.get_rng_state())


@pytest.mark.parametrize('change',['head','norm','attention_dropout'])
def test_initial_and_cached_model_guards_preserve_frozen_zero_attention_dropout(change):
    from solution.test_unpad import _tiny_peft_model
    from solution.unpad import _inspect,_layout
    base=_tiny_peft_model().get_base_model()
    assert _inspect(base) is not None
    assert _layout(base) is not None
    if change=='head':base.lm_head.weight.requires_grad_(True)
    elif change=='norm':base.model.layers[0].input_layernorm.weight.requires_grad_(True)
    else:base.model.layers[0].self_attn.attention_dropout=.1
    assert _inspect(base) is None
    assert _layout(base) is None

"""Actual Soup/TRL/PEFT AMP boundary: streaming is private, public logits Tensor."""
import inspect
from types import SimpleNamespace
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires Kaggle T4 CUDA')


def _trainer(model, accelerator):
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class
    from trl import SFTTrainer
    trainer=object.__new__(make_rewind_trainer_class(SFTTrainer))
    trainer.model,trainer.accelerator=model,accelerator
    trainer.args=SimpleNamespace(fp16=True,loss_type='nll',use_liger_kernel=False,
        packing=False,gradient_checkpointing=False,activation_offloading=False,
        torch_compile=False,average_tokens_across_devices=False,n_gpu=1)
    trainer.compute_loss_func=trainer.label_smoother=None
    trainer.model_accepts_loss_kwargs=True
    trainer.num_virtual_tokens=0
    trainer.aux_loss_enabled=trainer.padding_free=False
    trainer._total_train_tokens=0
    trainer._metrics={mode:{key:[] for key in ('entropy','num_tokens','mean_token_accuracy')}
                      for mode in ('train','eval')}
    records=[]
    def fail(exc):raise AssertionError(f'recorder failed: {exc}')
    trainer._soup_rewind_state=SimpleNamespace(failed=False,
        on_micro_batch=lambda **kw:records.append(kw),fail=fail)
    trainer.state=SimpleNamespace(global_step=0)
    return trainer,records


@pytest.mark.parametrize('return_outputs',[False,True])
def test_real_outer_metrics_lora_gradients_and_public_tensor_boundary(monkeypatch,return_outputs):
    from accelerate import Accelerator
    from transformers.models.qwen2 import modeling_qwen2
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf
    from solution.test_unpad import _tiny_peft_model,_batch
    from solution import private_scope,unpad
    # Each helper resets its own seed, including nonzero LoRA B weights.
    reference,candidate=_tiny_peft_model(),_tiny_peft_model()
    accelerator=Accelerator(mixed_precision='fp16')
    reference,candidate=accelerator.prepare_model(reference),accelerator.prepare_model(candidate)
    ref_trainer,ref_records=_trainer(reference,accelerator)
    trainer,records=_trainer(candidate,accelerator)
    ids,mask,labels=_batch([19,11],19,3001)
    inputs={'input_ids':ids,'attention_mask':mask,'labels':labels}
    items=(labels[:,1:]!=-100).sum()
    for owner,name in ((sft_trainer,'entropy_from_logits'),(rewind_hf,'row_losses')):
        monkeypatch.setattr(owner,name,getattr(owner,name))
    cls=modeling_qwen2.Qwen2ForCausalLM
    monkeypatch.setattr(cls,'forward',cls.forward)
    unpad.patch_qwen2_causal_lm()
    assert private_scope.wrap_trainer(trainer)
    seen=[];scales=[]
    permits=private_scope.permits
    def record(*args):
        valid=permits(*args);seen.append(valid)
        if valid:scales.append(private_scope.expected_loss_scale())
        return valid
    monkeypatch.setattr(private_scope,'permits',record)
    torch.manual_seed(1234)
    expected=ref_trainer.compute_loss(reference,dict(inputs),return_outputs=return_outputs,
                                     num_items_in_batch=items)
    torch.manual_seed(1234)
    actual=trainer.compute_loss(candidate,dict(inputs),return_outputs=return_outputs,
                                num_items_in_batch=items)
    assert private_scope._SCOPE.get() is None
    if return_outputs:
        expected_loss,expected_output=expected;actual_loss,actual_output=actual
        assert torch.is_tensor(actual_output.logits)
        assert actual_output.logits.dtype==expected_output.logits.dtype==torch.float32
        torch.testing.assert_close(actual_output.logits,expected_output.logits,rtol=0,atol=0)
        assert not any(seen), 'public output request must use original Tensor forward'
    else:
        expected_loss,actual_loss=expected,actual
        assert any(seen), 'private streaming route not exercised'
        assert scales and all(isinstance(s,(int,float)) and s>0 for s in scales)
    torch.testing.assert_close(actual_loss,expected_loss,rtol=2e-3,atol=2e-3)
    for key in ('entropy','num_tokens','mean_token_accuracy'):
        torch.testing.assert_close(torch.tensor(trainer._metrics['train'][key]),
                                   torch.tensor(ref_trainer._metrics['train'][key]),
                                   rtol=2e-3,atol=2e-3)
    assert len(records)==len(ref_records)==1
    assert records[0]['row_tokens']==ref_records[0]['row_tokens']
    torch.testing.assert_close(torch.tensor(records[0]['row_loss']),
                               torch.tensor(ref_records[0]['row_loss']),rtol=2e-3,atol=2e-3)
    (actual_loss*65536).backward();(expected_loss*65536).backward()
    for (name,p),(other_name,q) in zip(candidate.named_parameters(),reference.named_parameters()):
        assert name==other_name
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            # Unpadding changes cuBLAS M dimensions. Use the same 2% relative
            # L2 criterion as the teammate's original full-model comparison.
            relative=(p.grad.float()-q.grad.float()).norm()/q.grad.float().norm().clamp(min=1e-6)
            assert bool(torch.isfinite(relative)) and float(relative)<2e-2
    public=candidate(**dict(inputs),use_cache=False)
    assert torch.is_tensor(public.logits) and public.logits.dtype==torch.float32


def test_scope_restores_after_exception_and_binds_exact_objects():
    from solution import private_scope
    model=object();ids=torch.zeros((1,2),device='cuda',dtype=torch.int64)
    scope=private_scope._Scope(model,ids,ids,ids)
    token=private_scope._SCOPE.set(scope)
    try:
        assert private_scope.permits(model,ids,ids,ids)
        assert not private_scope.permits(object(),ids,ids,ids)
        assert not private_scope.permits(model,ids.clone(),ids,ids)
    finally:private_scope._SCOPE.reset(token)
    assert private_scope._SCOPE.get() is None

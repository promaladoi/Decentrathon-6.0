"""Validate the joint memory policy and native/private model boundaries."""
import pytest
import torch


def test_storage_wrapper_preserves_private_marker_and_delegation(monkeypatch):
    from types import SimpleNamespace
    from transformers.models.qwen2 import modeling_qwen2
    from solution.frozen_storage import patch_frozen_storage
    from solution.test_frozen_storage import pair
    embedding, head = pair()
    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding), lm_head=head)
    calls = []
    failure = RuntimeError('original failure')

    def native(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        if kwargs.get('fail'):
            raise failure
        return head

    native._soup_private_unpad = True
    cls = modeling_qwen2.Qwen2ForCausalLM
    monkeypatch.setattr(cls, 'forward', native)
    patch_frozen_storage()
    wrapper = cls.forward
    assert wrapper._soup_private_unpad is True
    assert wrapper(model, 17, labels='same') is head
    assert calls == [(model, (17,), {'labels': 'same'})]
    with pytest.raises(RuntimeError) as exc:
        wrapper(model, fail=True)
    assert exc.value is failure and len(calls) == 2
    patch_frozen_storage()
    assert cls.forward is wrapper


def test_packed_pair_needs_no_additional_private_head_cast():
    from solution.test_frozen_storage import pair
    from solution.frozen_storage import install_frozen_storage
    from solution.head_weight_cache import get_half_weight, clear_half_weight_cache
    embedding, head = pair()
    clear_half_weight_cache()
    original = head.weight.detach().clone()
    before = head.weight.numel() * head.weight.element_size()
    state = install_frozen_storage(embedding, head)
    try:
        private_weight = get_half_weight(head.weight)
        assert private_weight is head.weight is embedding.weight
        assert private_weight.untyped_storage().nbytes() == before // 2
        assert state['storage_bytes_saved'] == before // 2
        torch.testing.assert_close(private_weight.float(), original, rtol=0, atol=0)
        # A previously cached FP32 source must not be retained by this policy.
        from solution import head_weight_cache
        assert head_weight_cache._entry is None
    finally:
        clear_half_weight_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires T4/CUDA model integration')
@pytest.mark.parametrize('return_outputs', [False, True])
def test_lossless_storage_with_native_trainer_metrics_and_lora_gradients(monkeypatch, return_outputs):
    from solution import test_unpad, test_private_streaming
    from solution.frozen_storage import install_frozen_storage
    original_factory = test_unpad._tiny_peft_model
    constructed = []

    def model_factory():
        model = original_factory()
        base = model.get_base_model()
        embedding, head = base.model.embed_tokens, base.lm_head
        assert head.weight is embedding.weight and not head.weight.requires_grad
        # Reproduce the fixed model's fp16 checkpoint -> fp32 k-bit preparation.
        with torch.no_grad():
            head.weight.data = head.weight.data.half().float()
        if constructed:
            assert len(constructed) == 1
            assert install_frozen_storage(embedding, head) is not None
            assert head.weight.dtype == torch.float16
        else:
            assert head.weight.dtype == torch.float32
        constructed.append(model)
        return model

    monkeypatch.setattr(test_unpad, '_tiny_peft_model', model_factory)
    # Reuse the unchanged teammate test's native Soup/TRL/Accelerate recorder,
    # metrics, public Tensor output and all-LoRA-gradient assertions. No looser
    # numerical tolerances are introduced for the storage integration.
    test_private_streaming.test_real_outer_metrics_lora_gradients_and_public_tensor_boundary(
        monkeypatch, return_outputs)
    assert len(constructed) == 2

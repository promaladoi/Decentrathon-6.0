from types import SimpleNamespace
import pytest
import torch


def test_early_storage_ignores_unknown_model():
    from solution.frozen_storage import prepare_trainer_storage
    assert prepare_trainer_storage(SimpleNamespace(model=object())) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA model storage')
@pytest.mark.parametrize('observed', [False, 'head', 'model'])
def test_early_storage_preserves_values_ties_and_observers(observed):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from solution.frozen_storage import prepare_trainer_storage
    config = Qwen2Config(vocab_size=2048, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).half().cuda().requires_grad_(False)
    model.lm_head.weight.data = model.lm_head.weight.data.float()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    weight = model.lm_head.weight
    reference = weight.detach().clone()
    ids = torch.tensor([0, 18, 2047], device='cuda')
    values = torch.randn((3, 32), device='cuda')
    embedding_ref = model.model.embed_tokens(ids)
    head_ref = model.lm_head(values)
    observer = model if observed == 'model' else model.lm_head
    handle = observer.register_forward_pre_hook(lambda *args: None) if observed else None
    try:
        state = prepare_trainer_storage(SimpleNamespace(model=model))
        if observed:
            assert state is None and weight.dtype == torch.float32
        else:
            assert state['active'] and weight.dtype == torch.float16
            assert model._solution_storage_attempted
            assert prepare_trainer_storage(SimpleNamespace(model=model)) is state
            assert model.lm_head.weight is model.model.embed_tokens.weight is weight
        torch.testing.assert_close(weight.float(), reference, rtol=0, atol=0)
        torch.testing.assert_close(model.model.embed_tokens(ids), embedding_ref, rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head(values), head_ref, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()

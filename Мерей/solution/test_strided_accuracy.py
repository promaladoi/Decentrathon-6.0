"""Strided accuracy semantics and preservation of the pinned method closure.

The CUDA full Soup/TRL/PEFT combination is exercised by test_private_logits.
"""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cpu_strided_argmax_matches_copy_and_promoted_values(dtype):
    # Two batches make the sequence-truncated view noncontiguous. Include
    # ties, infinities, and multiple NaNs to exercise first-index semantics.
    logits = torch.zeros((2, 5, 17), dtype=dtype)
    logits[0, 0, 2:4] = 7
    logits[0, 1, :] = -float("inf")
    logits[0, 2, 6] = logits[0, 2, 8] = float("nan")
    logits[1, 0, 4] = logits[1, 0, 9] = float("inf")
    view = logits[..., :-1, :]
    assert not view.is_contiguous()
    actual = view.argmax(dim=-1)
    torch.testing.assert_close(actual, view.contiguous().argmax(dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(actual, view.float().contiguous().argmax(dim=-1), rtol=0, atol=0)
    assert actual[0, 0].item() == 2 and actual[0, 1].item() == 0
    assert actual[0, 2].item() == 6 and actual[1, 0].item() == 4


def test_clone_reuses_original_globals_and_class_closure():
    from trl import SFTTrainer
    from solution import strided_accuracy

    original = inspect.unwrap(SFTTrainer.compute_loss)
    cloned = strided_accuracy._clone_pinned_method(original, SFTTrainer)
    assert cloned.__globals__ is original.__globals__
    assert cloned.__closure__ is original.__closure__
    assert cloned.__code__.co_freevars == original.__code__.co_freevars == ("__class__",)
    assert inspect.signature(cloned) == inspect.signature(original)


def test_mismatched_version_fails_closed(monkeypatch):
    from trl import SFTTrainer
    from solution import strided_accuracy

    original = inspect.unwrap(SFTTrainer.compute_loss)
    monkeypatch.setattr(SFTTrainer, "compute_loss", original)
    monkeypatch.setattr(strided_accuracy.importlib.metadata, "version", lambda name: "0.0.0")
    assert not strided_accuracy.install()
    assert SFTTrainer.compute_loss is original
    assert "version differs" in strided_accuracy.guard_reason()


def test_cpu_cloned_full_metric_method_matches_original():
    from accelerate.utils import operations
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from trl import SFTTrainer
    from solution import strided_accuracy
    from solution.test_private_logits import _small_real_rewind_trainer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(2, 5, 17))
        def forward(self, input_ids, labels, attention_mask, use_cache=False,
                    num_items_in_batch=None):
            return CausalLMOutputWithPast(
                loss=ForCausalLMLoss(self.logits, labels, 17, num_items_in_batch),
                logits=self.logits)

    torch.manual_seed(231)
    reference = Model()
    candidate = copy.deepcopy(reference)
    accelerator = SimpleNamespace(parallelism_config=None, num_processes=1,
                                  gather_for_metrics=lambda value: value,
                                  unwrap_model=lambda model: model)
    ref_trainer, _ = _small_real_rewind_trainer(reference, accelerator)
    trainer, _ = _small_real_rewind_trainer(candidate, accelerator)
    original = inspect.unwrap(SFTTrainer.compute_loss)
    cloned = strided_accuracy._clone_pinned_method(original, SFTTrainer)
    labels = torch.randint(17, (2, 5))
    labels[:, :2] = -100
    inputs = {"input_ids": labels.clamp(min=0), "labels": labels,
              "attention_mask": torch.ones_like(labels)}
    expected_loss, expected_output = original(
        ref_trainer, reference, dict(inputs), return_outputs=True, num_items_in_batch=6)
    actual_loss, actual_output = cloned(
        trainer, candidate, dict(inputs), return_outputs=True, num_items_in_batch=6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_output.logits, expected_output.logits, rtol=0, atol=0)
    assert trainer._metrics == ref_trainer._metrics
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(candidate.logits.grad, reference.logits.grad, rtol=0, atol=0)

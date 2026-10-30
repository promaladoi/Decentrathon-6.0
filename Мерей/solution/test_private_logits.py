"""Public output boundaries and private metric lifetime checks."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def _cpu_plumbing(monkeypatch, base_class):
    """Exercise the real wrapper boundary, stubbing only CUDA eligibility."""
    from accelerate.utils import operations
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class
    from solution import private_logits

    original_convert = operations.convert_to_fp32
    monkeypatch.setattr(operations, "convert_to_fp32", original_convert)
    monkeypatch.setattr(private_logits, "_ORIGINAL_CONVERT", None)
    monkeypatch.setattr(private_logits, "_can_keep_logits",
                        lambda output, scope: scope is not None)
    labels = torch.zeros((1, 2), dtype=torch.int64)
    scope = private_logits._Scope(labels, (1, 2, 3))
    monkeypatch.setattr(private_logits, "_scope_for_call", lambda *args: (
        None if private_logits.private_logits_active() else scope))
    private_logits._install_output_conversion()
    trainer = make_rewind_trainer_class(base_class)()
    assert private_logits.wrap_trainer(trainer)
    assert not private_logits.wrap_trainer(trainer)
    return private_logits, trainer, original_convert


@pytest.mark.parametrize("return_outputs", [False, True])
def test_cpu_private_boundary_restores_public_fp32(monkeypatch, return_outputs):
    from accelerate.utils import operations
    from transformers.modeling_outputs import CausalLMOutputWithPast

    observed = []
    class Base:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logits = inputs["logits"]
            output = operations.convert_to_fp32(CausalLMOutputWithPast(
                loss=logits.float().square().mean(), logits=logits))
            observed.append(output.logits.dtype)
            return (output.loss, output) if return_outputs else output.loss

    private, trainer, original_convert = _cpu_plumbing(monkeypatch, Base)
    logits = torch.randn((1, 2, 3), dtype=torch.float16, requires_grad=True)
    result = trainer.compute_loss(object(), {"logits": logits}, return_outputs=return_outputs)
    assert observed == [torch.float16]
    assert not private.private_logits_active()
    assert list(inspect.signature(trainer.compute_loss).parameters) == [
        "model", "inputs", "return_outputs", "num_items_in_batch"]
    if return_outputs:
        loss, output = result
        assert type(output) is CausalLMOutputWithPast
        assert output.logits.dtype == torch.float32
        torch.testing.assert_close(output.logits, logits.float(), rtol=0, atol=0)
        loss = loss + output.logits.square().mean()
    else:
        loss = result
    assert loss.dtype == torch.float32
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == torch.float16
    outside = operations.convert_to_fp32(CausalLMOutputWithPast(logits=logits))
    expected = original_convert(CausalLMOutputWithPast(logits=logits))
    assert outside.logits.dtype == torch.float32
    torch.testing.assert_close(outside.logits, expected.logits, rtol=0, atol=0)


def test_cpu_private_scope_resets_after_exception(monkeypatch):
    class Base:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            from solution import private_logits
            assert private_logits.private_logits_active()
            raise RuntimeError("expected test exception")

    private, trainer, _ = _cpu_plumbing(monkeypatch, Base)
    with pytest.raises(RuntimeError, match="expected test exception"):
        trainer.compute_loss(object(), {})
    assert not private.private_logits_active()


def test_cpu_nested_call_suspends_and_restores_outer_scope(monkeypatch):
    from accelerate.utils import operations
    from transformers.modeling_outputs import CausalLMOutputWithPast

    seen = []
    class Base:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            from solution import private_logits
            seen.append(private_logits.private_logits_active())
            if inputs.get("outer"):
                _loss, nested_output = self.compute_loss(
                    model, {"outer": False}, return_outputs=True)
                assert nested_output.logits.dtype == torch.float32
                assert private_logits.private_logits_active()
            output = operations.convert_to_fp32(CausalLMOutputWithPast(
                loss=torch.tensor(1.0), logits=torch.zeros((1, 2, 3), dtype=torch.float16)))
            return (output.loss, output) if return_outputs else output.loss

    private, trainer, _ = _cpu_plumbing(monkeypatch, Base)
    trainer.compute_loss(object(), {"outer": True})
    assert seen == [True, False]
    assert not private.private_logits_active()


def test_forward_observers_are_detected():
    from solution import private_logits

    model = torch.nn.Linear(2, 2)
    assert not private_logits._has_output_observer(model)
    handle = model.register_forward_hook(lambda module, args, output: None)
    try:
        assert private_logits._has_output_observer(model)
    finally:
        handle.remove()
    assert not private_logits._has_output_observer(model)


def test_idempotent_converter_install_repairs_original_reference(monkeypatch):
    from accelerate.utils import operations
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from solution import private_logits

    monkeypatch.setattr(operations, "convert_to_fp32", operations.convert_to_fp32)
    monkeypatch.setattr(private_logits, "_ORIGINAL_CONVERT", private_logits._ORIGINAL_CONVERT)
    private_logits._install_output_conversion()
    installed = operations.convert_to_fp32
    original = installed._soup_private_logits_original
    private_logits._ORIGINAL_CONVERT = None
    assert not private_logits._install_output_conversion()
    assert operations.convert_to_fp32 is installed
    assert private_logits._ORIGINAL_CONVERT is original
    output = private_logits._ORIGINAL_CONVERT(CausalLMOutputWithPast(
        logits=torch.ones((1, 2, 3), dtype=torch.float16)))
    assert output.logits.dtype == torch.float32


def _small_peft_model():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32,
                         attention_dropout=0.0, use_cache=False)
    model = get_peft_model(Qwen2ForCausalLM(config).cuda(), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    return model


def _small_real_rewind_trainer(model, accelerator):
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class
    from trl import SFTTrainer

    # Bypass only dataset/tokenizer construction. compute_loss is the actual
    # pinned Soup -> TRL -> Transformers chain, using an actual AMP PEFT model.
    trainer = object.__new__(make_rewind_trainer_class(SFTTrainer))
    trainer.model, trainer.accelerator = model, accelerator
    trainer.args = SimpleNamespace(
        fp16=True, loss_type="nll", use_liger_kernel=False, packing=False,
        gradient_checkpointing=False, activation_offloading=False,
        torch_compile=False, average_tokens_across_devices=False, n_gpu=1)
    trainer.compute_loss_func = trainer.label_smoother = None
    trainer.model_accepts_loss_kwargs = True
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = trainer.padding_free = False
    trainer._total_train_tokens = 0
    trainer._metrics = {mode: {key: [] for key in
                        ("entropy", "num_tokens", "mean_token_accuracy")}
                        for mode in ("train", "eval")}
    records = []
    def fail(error):
        raise AssertionError(f"recorder failed: {error}")
    trainer._soup_rewind_state = SimpleNamespace(
        failed=False, on_micro_batch=lambda **kw: records.append(kw), fail=fail)
    trainer.state = SimpleNamespace(global_step=0)
    return trainer, records


@CUDA
@pytest.mark.parametrize("return_outputs", [False, True])
@pytest.mark.parametrize("use_strided_accuracy", [False, True])
def test_actual_outer_soup_metrics_outputs_and_lora_gradients(
        monkeypatch, return_outputs, use_strided_accuracy):
    from accelerate import Accelerator
    from accelerate.utils import operations
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from trl import SFTTrainer
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf
    from solution import half_loss, private_logits

    # Other integration tests may have installed the combined setup already.
    # Isolate this ablation and restore that pre-existing method afterwards.
    monkeypatch.setattr(SFTTrainer, "compute_loss", inspect.unwrap(SFTTrainer.compute_loss))
    # Restore every patched global after this test, including the lazy builder.
    for owner, name in ((operations, "convert_to_fp32"),
                        (SFTTrainerWrapper, "_build_rewind_trainer"),
                        (sft_trainer, "entropy_from_logits"),
                        (rewind_hf, "row_losses")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(private_logits, "_ORIGINAL_CONVERT", private_logits._ORIGINAL_CONVERT)
    monkeypatch.setattr(private_logits, "_HALF_METRICS", private_logits._HALF_METRICS)
    torch.manual_seed(221)
    reference = _small_peft_model()
    candidate = copy.deepcopy(reference)
    assert half_loss.install(reference) == half_loss.install(candidate) == 1
    accelerator = Accelerator(mixed_precision="fp16")
    reference = accelerator.prepare_model(reference)
    candidate = accelerator.prepare_model(candidate)
    ref_trainer, ref_records = _small_real_rewind_trainer(reference, accelerator)
    trainer, records = _small_real_rewind_trainer(candidate, accelerator)
    private_logits.install()
    if use_strided_accuracy:
        from solution import strided_accuracy
        assert strided_accuracy.install(), strided_accuracy.guard_reason()
    assert private_logits.wrap_trainer(trainer)
    observed = []
    can_keep = private_logits._can_keep_logits
    def record(output, scope):
        valid = can_keep(output, scope)
        observed.append(valid)
        return valid
    monkeypatch.setattr(private_logits, "_can_keep_logits", record)
    input_ids = torch.randint(3, 97, (2, 9), device="cuda")
    labels = input_ids.clone()
    labels[:, :3] = -100
    inputs = {"input_ids": input_ids, "labels": labels,
              "attention_mask": torch.ones_like(input_ids)}
    torch.manual_seed(222)
    expected = ref_trainer.compute_loss(reference, dict(inputs),
                                        return_outputs=return_outputs,
                                        num_items_in_batch=torch.tensor(12, device="cuda"))
    torch.manual_seed(222)
    actual = trainer.compute_loss(candidate, dict(inputs), return_outputs=return_outputs,
                                  num_items_in_batch=torch.tensor(12, device="cuda"))
    assert observed and any(observed), "private conversion was not exercised"
    assert not private_logits.private_logits_active()
    if return_outputs:
        expected_loss, expected_output = expected
        actual_loss, actual_output = actual
        assert actual_output.logits.dtype == expected_output.logits.dtype == torch.float32
        torch.testing.assert_close(actual_output.logits, expected_output.logits, rtol=0, atol=0)
    else:
        actual_loss, expected_loss = actual, expected
    torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-6, atol=3e-6)
    for key in ("entropy", "num_tokens", "mean_token_accuracy"):
        torch.testing.assert_close(torch.tensor(trainer._metrics["train"][key]),
                                   torch.tensor(ref_trainer._metrics["train"][key]),
                                   rtol=2e-5, atol=3e-6)
    assert len(records) == len(ref_records) == 1
    assert records[0]["step"] == ref_records[0]["step"]
    assert records[0]["row_tokens"] == ref_records[0]["row_tokens"]
    torch.testing.assert_close(torch.tensor(records[0]["row_loss"]),
                               torch.tensor(ref_records[0]["row_loss"]),
                               rtol=1e-6, atol=1e-6)
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (other_name, other) in zip(
            candidate.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and other.grad is not None
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-3, atol=2e-5)
    # Public model calls after the private scope retain the original fp32 API.
    assert candidate(**dict(inputs), use_cache=False).logits.dtype == torch.float32

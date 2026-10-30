"""Experimental half-input loss: cast parity, wrapper normalization and AMP."""

import copy

import pytest
import torch
import torch.nn.functional as F


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def test_half_loss_install_cpu_model_is_noop():
    from solution import half_loss

    assert half_loss.install(torch.nn.Linear(2, 2)) == 0
    assert half_loss.install(object()) == 0


@CUDA
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("ignore_index", [-100, -17])
@pytest.mark.parametrize("rows,width", [(7, 4103), (3, 151936)])
def test_half_ce_matches_explicit_float_upcast(reduction, ignore_index, rows, width):
    from solution.vocab import _half_cross_entropy

    torch.manual_seed(151)
    logits = (torch.randn((rows, width), device="cuda", dtype=torch.float16) * 3).requires_grad_(True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(width, (rows,), device="cuda")
    labels[1] = ignore_index
    labels[-1] = width - 1
    actual = _half_cross_entropy(logits, labels, ignore_index, reduction)
    expected = F.cross_entropy(reference.float(), labels,
                               ignore_index=ignore_index, reduction=reduction)
    upstream = torch.full_like(expected, 0.37)
    actual.backward(upstream)
    expected.backward(upstream)
    assert actual.dtype == expected.dtype == torch.float32
    assert logits.grad.dtype == reference.grad.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-5, atol=3e-7)


@CUDA
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_half_ce_all_ignored_preserves_nan_and_zero_gradient(reduction):
    from solution.vocab import _half_cross_entropy

    logits = torch.randn((5, 4103), device="cuda", dtype=torch.float16, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.full((5,), -17, device="cuda", dtype=torch.int64)
    actual = _half_cross_entropy(logits, labels, -17, reduction)
    expected = F.cross_entropy(reference.float(), labels, ignore_index=-17, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)
    assert torch.count_nonzero(logits.grad).item() == 0


def _model():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32,
                         attention_dropout=0.0, use_cache=False)
    return Qwen2ForCausalLM(config).cuda()


@CUDA
@pytest.mark.parametrize("normalizer", ["none", "integer", "cpu_tensor", "device_tensor"])
def test_instance_loss_preserves_hf_num_items_and_uses_half_kernel(monkeypatch, normalizer):
    from solution import half_loss, vocab

    torch.manual_seed(152)
    model = _model()
    original = model.loss_function
    assert half_loss.install(model) == 1
    first = model.loss_function
    assert half_loss.install(model) == 0 and model.loss_function is first
    logits = torch.randn((2, 9, 97), device="cuda", dtype=torch.float16, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(97, (2, 9), device="cuda")
    labels[:, :3] = -100
    denominator = {"none": None, "integer": 19,
                   "cpu_tensor": torch.tensor(19),
                   "device_tensor": torch.tensor(19, device="cuda")}[normalizer]
    expected = original(reference, labels, 97, denominator)
    expected.backward()
    observed = []
    run = vocab.vocabulary_partials_kernel.run

    def record(*args, **kwargs):
        observed.append(args[0].dtype)
        return run(*args, **kwargs)

    monkeypatch.setattr(vocab.vocabulary_partials_kernel, "run", record)
    actual = model.loss_function(logits, labels, 97, denominator)
    actual.backward()
    assert observed and all(dtype == torch.float16 for dtype in observed)
    assert logits.grad.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-5, atol=3e-7)


@CUDA
def test_real_fp16_qwen_accelerate_outputs_and_gradients_match(monkeypatch):
    from accelerate import Accelerator
    from solution import half_loss, vocab

    torch.manual_seed(153)
    reference = _model()
    candidate = copy.deepcopy(reference)
    assert half_loss.install(candidate) == 1
    accelerator = Accelerator(mixed_precision="fp16")
    reference = accelerator.prepare_model(reference)
    candidate = accelerator.prepare_model(candidate)
    input_ids = torch.randint(3, 97, (2, 9), device="cuda")
    labels = input_ids.clone()
    labels[:, :3] = -100
    expected = reference(input_ids=input_ids, labels=labels)
    expected.loss.backward()
    observed = []
    run = vocab.vocabulary_gradient_kernel.run

    def record(*args, **kwargs):
        observed.append((args[0].dtype, args[5].dtype))
        return run(*args, **kwargs)

    monkeypatch.setattr(vocab.vocabulary_gradient_kernel, "run", record)
    actual = candidate(input_ids=input_ids, labels=labels)
    actual.loss.backward()
    assert observed and all(pair == (torch.float16, torch.float16) for pair in observed)
    # Accelerate still performs its usual output conversion. No logits patch.
    assert actual.logits.dtype == expected.logits.dtype == torch.float32
    assert actual.loss.dtype == expected.loss.dtype == torch.float32
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=3e-6, atol=3e-6)
    for (name, parameter), (reference_name, ref_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        if parameter.requires_grad:
            assert parameter.grad is not None and ref_parameter.grad is not None
            torch.testing.assert_close(parameter.grad, ref_parameter.grad,
                                       rtol=8e-4, atol=2e-5)


@CUDA
def test_instance_hook_preserves_custom_loss():
    from solution import half_loss

    model = _model()
    sentinel = lambda *args, **kwargs: None
    model.loss_function = sentinel
    assert half_loss.install(model) == 0
    assert model.loss_function is sentinel


@CUDA
@pytest.mark.parametrize("provide_labels", [False, True])
def test_explicit_shift_labels_delegates_original_arguments(monkeypatch, provide_labels):
    from transformers.loss.loss_utils import ForCausalLMLoss
    from solution import half_loss, vocab

    torch.manual_seed(154)
    logits = torch.randn((2, 5, 97), device="cuda", dtype=torch.float16,
                         requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(97, (2, 5), device="cuda") if provide_labels else None
    shifted = torch.randint(97, (2, 5), device="cuda")
    shifted[:, -1] = -100
    denominator = torch.tensor(11)
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return ForCausalLMLoss(*args, **kwargs)

    def forbidden_half_path(*args, **kwargs):
        raise AssertionError("explicit shift_labels must use the original loss")

    monkeypatch.setattr(vocab, "_half_cross_entropy", forbidden_half_path)
    expected = ForCausalLMLoss(reference, labels, 97, denominator,
                               shift_labels=shifted, unused_flag="preserved")
    actual = half_loss._wrap_loss(original)(logits, labels, 97, denominator,
                                            shift_labels=shifted,
                                            unused_flag="preserved")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is logits and args[1] is labels and args[3] is denominator
    assert args[5] is shifted and kwargs == {"unused_flag": "preserved"}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("all_ignored", [False, True])
def test_zero_num_items_preserves_nonfinite_loss_and_gradient(all_ignored):
    from solution import half_loss

    torch.manual_seed(155)
    model = _model()
    original = model.loss_function
    assert half_loss.install(model) == 1
    logits = torch.randn((2, 5, 97), device="cuda", dtype=torch.float16,
                         requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.full((2, 5), -100, device="cuda", dtype=torch.int64)
    if not all_ignored:
        labels[:, 2:] = torch.randint(97, (2, 3), device="cuda")
    denominator = torch.tensor(0)
    expected = original(reference, labels, 97, denominator)
    actual = model.loss_function(logits, labels, 97, denominator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if all_ignored:
        assert torch.isnan(actual).item()
    else:
        assert torch.isposinf(actual).item()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad,
                               rtol=0, atol=0, equal_nan=True)
    if all_ignored:
        assert torch.count_nonzero(logits.grad).item() == 0


@CUDA
def test_extra_loss_kwargs_remain_ignored_on_half_fast_path(monkeypatch):
    from solution import half_loss, vocab

    torch.manual_seed(156)
    model = _model()
    original = model.loss_function
    assert half_loss.install(model) == 1
    logits = torch.randn((2, 5, 97), device="cuda", dtype=torch.float16,
                         requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(97, (2, 5), device="cuda")
    # HF fixed_cross_entropy ignores **kwargs, including names accepted by
    # torch.nn.functional.cross_entropy. They must not change its fixed task.
    kwargs = {"reduction": "none", "label_smoothing": 0.3,
              "weight": torch.rand(97, device="cuda"), "unknown_option": 17,
              "output_attentions": None, "output_hidden_states": None}
    expected = original(reference, labels, 97, **kwargs)
    calls = []
    run = vocab._half_cross_entropy

    def record(source, *args, **kw):
        calls.append(source.dtype)
        return run(source, *args, **kw)

    monkeypatch.setattr(vocab, "_half_cross_entropy", record)
    actual = model.loss_function(logits, labels, 97, **kwargs)
    assert actual.ndim == expected.ndim == 0
    assert calls == [torch.float16]
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-5, atol=3e-7)


@CUDA
def test_half_ce_keeps_amp_scale_in_float32_until_gradient_store():
    from solution.vocab import _half_cross_entropy

    torch.manual_seed(157)
    logits = torch.randn((4, 4103), device="cuda", dtype=torch.float16,
                         requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(4103, (4,), device="cuda")
    labels[0] = -100
    actual = _half_cross_entropy(logits, labels)
    expected = F.cross_entropy(reference.float(), labels)
    # A common GradScaler initial scale exceeds the finite fp16 range. The
    # upstream scalar must stay fp32 until the complete scaled gradient casts.
    upstream = torch.full_like(expected, 65536.0)
    assert upstream.dtype == torch.float32 and torch.isfinite(upstream).item()
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    assert torch.isfinite(logits.grad).all().item()
    # Permit one fp16 rounding step at scaled magnitudes, not an fp16 scale.
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-3, atol=5e-4)


@CUDA
def test_actual_peft_lora_amp_outputs_and_adapter_gradients_match():
    from accelerate import Accelerator
    from peft import LoraConfig, TaskType, get_peft_model
    from solution import half_loss

    torch.manual_seed(158)
    reference = get_peft_model(_model(), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM))
    # Nonzero B matrices exercise gradients of both LoRA factors on step one.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    candidate = copy.deepcopy(reference)
    original = reference.get_base_model().loss_function
    assert half_loss.install(candidate) == 1
    assert half_loss.install(candidate) == 0
    assert reference.get_base_model().loss_function is original
    assert getattr(candidate.get_base_model().loss_function,
                   "_soup_half_loss_patch", False)
    accelerator = Accelerator(mixed_precision="fp16")
    reference = accelerator.prepare_model(reference)
    candidate = accelerator.prepare_model(candidate)
    input_ids = torch.randint(3, 97, (2, 9), device="cuda")
    labels = input_ids.clone()
    labels[:, :3] = -100
    # Preserve real LoRA dropout and replay its RNG rather than disabling it.
    torch.manual_seed(159)
    expected = reference(input_ids=input_ids, labels=labels,
                         num_items_in_batch=torch.tensor(12, device="cuda"))
    expected.loss.backward()
    torch.manual_seed(159)
    actual = candidate(input_ids=input_ids, labels=labels,
                       num_items_in_batch=torch.tensor(12, device="cuda"))
    actual.loss.backward()
    assert actual.logits.dtype == expected.logits.dtype == torch.float32
    assert actual.loss.dtype == expected.loss.dtype == torch.float32
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=3e-6, atol=3e-6)
    compared = []
    gradient_signal = False
    for (name, parameter), (reference_name, ref_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        if parameter.requires_grad:
            assert "lora_" in name
            assert parameter.grad is not None and ref_parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()
            torch.testing.assert_close(parameter.grad, ref_parameter.grad,
                                       rtol=1e-3, atol=2e-5)
            compared.append(name)
            gradient_signal |= bool(torch.count_nonzero(parameter.grad).item())
        else:
            assert parameter.grad is None and ref_parameter.grad is None
    assert compared and all("lora_" in name for name in compared)
    assert gradient_signal

"""Exercise real call sites and fallbacks without model/data downloads."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch


def _device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        pytest.importorskip("triton")
    return torch.device(name)


def _loss_module(monkeypatch):
    loss_utils = pytest.importorskip("transformers.loss.loss_utils")
    original = inspect.unwrap(loss_utils.fixed_cross_entropy)
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    return loss_utils, original


def _output_modules(monkeypatch):
    sft = pytest.importorskip("trl.trainer.sft_trainer")
    rewind = pytest.importorskip("soup_cli.trainer.rewind_hf")
    original_entropy = inspect.unwrap(sft.entropy_from_logits)
    original_rows = inspect.unwrap(rewind.row_losses)
    monkeypatch.setattr(sft, "entropy_from_logits", original_entropy)
    monkeypatch.setattr(rewind, "row_losses", original_rows)
    return sft, rewind, original_entropy, original_rows


@pytest.mark.parametrize("kind", ["absent", "fake", "module"])
def test_apply_is_idempotent_and_guards_fake_soup_models(monkeypatch, kind):
    from solution import apply, patches
    sft = pytest.importorskip("soup_cli.trainer.sft")

    calls = []
    sentinel = object()

    def build(self, argument):
        assert argument == "input"
        calls.append("build")
        if kind == "fake":
            self.model = SimpleNamespace(config=SimpleNamespace())
        elif kind == "module":
            self.model = torch.nn.Linear(2, 2)
        return sentinel

    monkeypatch.setattr(sft.SFTTrainerWrapper, "_setup_transformers", build)
    monkeypatch.setattr(patches, "install_model_patches", lambda: calls.append("install"))
    apply()
    first = sft.SFTTrainerWrapper._setup_transformers
    apply()
    assert sft.SFTTrainerWrapper._setup_transformers is first
    wrapper = object.__new__(sft.SFTTrainerWrapper)
    assert wrapper._setup_transformers("input") is sentinel
    assert calls == (["build", "install"] if kind == "module" else ["build"])


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("normalizer", ["none", "integer", "cpu_tensor", "device_tensor"])
def test_fixed_cross_entropy_preserves_normalization_and_gradients(
        monkeypatch, device_name, normalizer):
    from solution import patches

    device = _device(device_name)
    loss_utils, original = _loss_module(monkeypatch)
    patches.install_loss_patch()
    first = loss_utils.fixed_cross_entropy
    patches.install_loss_patch()
    assert loss_utils.fixed_cross_entropy is first
    torch.manual_seed(83)
    logits = torch.randn((7, 4103), device=device, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 1, -100, 4102, 3, -100, 77], device=device)
    count = {"none": None, "integer": 11,
             "cpu_tensor": torch.tensor(11),
             "device_tensor": torch.tensor(11, device=device)}[normalizer]
    actual = loss_utils.fixed_cross_entropy(logits, targets, count)
    expected = original(ref_logits, targets, count)
    (actual * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-5, atol=3e-7)


def test_fixed_cross_entropy_cpu_and_soft_targets_keep_eager_behavior(monkeypatch):
    from solution import patches

    loss_utils, original = _loss_module(monkeypatch)
    patches.install_loss_patch()
    torch.manual_seed(84)
    logits = torch.randn((4, 13), requires_grad=True)
    targets = torch.softmax(torch.randn_like(logits), dim=-1)
    actual, expected = loss_utils.fixed_cross_entropy(logits, targets), original(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_entropy_masks_and_rewind_rows_preserve_outputs(monkeypatch, device_name):
    from solution import patches

    device = _device(device_name)
    sft, rewind, entropy_original, rows_original = _output_modules(monkeypatch)
    patches.install_entropy_patch()
    patches.install_rewind_patch()
    first_entropy, first_rows = sft.entropy_from_logits, rewind.row_losses
    patches.install_entropy_patch()
    patches.install_rewind_patch()
    assert sft.entropy_from_logits is first_entropy
    assert rewind.row_losses is first_rows

    torch.manual_seed(85)
    logits = torch.randn((3, 9, 4103), device=device)
    labels = torch.randint(4103, (3, 9), device=device)
    labels[0, :4] = -100
    labels[1, :] = -100
    labels[2, -2:] = -100
    # TRL entropy uses the full attention mask, including prompt positions;
    # supervised-token masks are used for accuracy and row losses separately.
    attention_mask = torch.ones_like(labels)
    attention_mask[2, -2:] = 0
    with torch.no_grad():
        actual_entropy = sft.entropy_from_logits(logits)
        expected_entropy = entropy_original(logits)
        torch.testing.assert_close((actual_entropy * attention_mask).sum() / attention_mask.sum(),
                                   (expected_entropy * attention_mask).sum() / attention_mask.sum(),
                                   rtol=2e-5, atol=5e-6)
        actual_values, actual_counts = rewind.row_losses(logits, labels)
        expected_values, expected_counts = rows_original(logits, labels)
    assert actual_counts == expected_counts
    assert actual_counts[1] == 0 and actual_values[1] == 0.0
    torch.testing.assert_close(torch.tensor(actual_values), torch.tensor(expected_values),
                               rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_entropy_retains_gradient_when_requested(monkeypatch, device_name):
    from solution import patches

    device = _device(device_name)
    sft, _, entropy_original, _ = _output_modules(monkeypatch)
    patches.install_entropy_patch()
    torch.manual_seed(86)
    logits = torch.randn((2, 5, 17), device=device, requires_grad=True)
    actual, expected = sft.entropy_from_logits(logits), entropy_original(logits)
    assert actual.requires_grad
    actual_grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_tiny_qwen_training_uses_patches_after_soup_setup(monkeypatch, device_name, mixed_precision):
    """A random tiny PEFT Qwen, real loss/backward, and no network or checkpoints."""
    from solution import apply

    device = _device(device_name)
    if mixed_precision and device.type != "cuda":
        pytest.skip("this mixed precision contract is CUDA fp16")
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    peft = pytest.importorskip("peft")
    sft_wrapper = pytest.importorskip("soup_cli.trainer.sft")
    sft, rewind, _, _ = _output_modules(monkeypatch)
    _loss_module(monkeypatch)
    monkeypatch.setattr(modeling.Qwen2RMSNorm, "forward",
                        inspect.unwrap(modeling.Qwen2RMSNorm.forward))
    monkeypatch.setattr(modeling.Qwen2MLP, "forward",
                        inspect.unwrap(modeling.Qwen2MLP.forward))
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb",
                        inspect.unwrap(modeling.apply_rotary_pos_emb))
    torch.manual_seed(87)
    config = modeling.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=32,
                                 attention_dropout=0.0, use_cache=False)
    reference = peft.get_peft_model(
        modeling.Qwen2ForCausalLM(config),
        peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    ).to(device)
    model = copy.deepcopy(reference)
    input_ids = torch.randint(3, config.vocab_size, (2, 9), device=device)
    labels = input_ids.clone()
    labels[:, :3] = -100

    def forward(current_model):
        # This is Trainer/Accelerate's contract: fp16 model execution and
        # float32 returned logits/loss. Keep the metric call sites untouched.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=mixed_precision):
            return current_model(input_ids=input_ids, labels=labels)

    if mixed_precision:
        from accelerate.utils import convert_outputs_to_fp32
        forward = convert_outputs_to_fp32(forward)

    reference.train()
    expected = forward(reference)
    expected.loss.backward()

    # Exercise the actual after-construction hook with an nn.Module, instead
    # of directly installing model patches from the test.
    token = object()

    def build(self):
        self.model = model
        return token

    monkeypatch.setattr(sft_wrapper.SFTTrainerWrapper, "_setup_transformers", build)
    counts = {}
    if device.type == "cuda":
        from solution import rmsnorm, vocab, rope
        kernels = [rmsnorm.soup_rmsnorm_forward_kernel,
                   rmsnorm.soup_rmsnorm_backward_kernel,
                   vocab.vocabulary_partials_kernel, vocab.vocabulary_finish_kernel,
                   vocab.vocabulary_gradient_kernel,
                   rope.soup_rope_forward_kernel, rope.soup_rope_backward_kernel]
        for kernel in kernels:
            name, original_run = kernel.fn.__name__, kernel.run

            def record(*args, _name=name, _run=original_run, **kwargs):
                counts[_name] = counts.get(_name, 0) + 1
                return _run(*args, **kwargs)

            monkeypatch.setattr(kernel, "run", record)
    apply()
    wrapper = object.__new__(sft_wrapper.SFTTrainerWrapper)
    assert wrapper._setup_transformers() is token
    assert wrapper.model is model
    model.train()
    actual = forward(model)
    actual.loss.backward()
    if mixed_precision:
        # These fp32-returned logits were produced by fp16 GEMMs. In
        # particular, the unchanged starter SwiGLU rounds once instead of
        # rounding silu before multiplying. Use a half-precision model
        # tolerance here, while keeping loss, gradients and float32 strict.
        delta = (actual.logits.detach() - expected.logits.detach()).float()
        print({"tiny_qwen_fp16_logits_max_abs": delta.abs().max().item(),
               "tiny_qwen_fp16_logits_rms": delta.square().mean().sqrt().item(),
               "tiny_qwen_fp16_loss_abs":
                   (actual.loss.detach() - expected.loss.detach()).abs().item()})
        torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-3, atol=3e-4)
    else:
        torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=3e-6)
    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(),
                                                  reference.named_parameters()):
        assert name == ref_name
        if param.requires_grad:
            assert param.grad is not None and ref_param.grad is not None
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=8e-4, atol=2e-5)
    with torch.no_grad():
        metric_logits = actual.logits
        entropy = sft.entropy_from_logits(metric_logits)
        row_values, counts_rows = rewind.row_losses(metric_logits, labels)
    assert entropy.shape == input_ids.shape
    assert len(row_values) == 2 and counts_rows == [6, 6]
    if device.type == "cuda":
        for kernel in kernels:
            assert counts.get(kernel.fn.__name__, 0) > 0, kernel.fn.__name__

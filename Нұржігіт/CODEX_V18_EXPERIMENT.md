# v18: lossless frozen tied-weight storage (not GPU validated)

Built on the pending v17 decoder. PEFT's k-bit preparation upcasts ordinary
half-precision parameters to fp32. On this frozen LoRA task the tied embedding
and output head do not receive optimizer updates. This experiment stores their
same Parameter in fp16 only after checking **every value** survives the round
trip exactly and is finite. The check and compact copy share one Triton pass.

This is a storage change, not another quantization of the model. Original fp32
embedding outputs are restored. The ordinary non-autocast head reconstructs
fp32 weights temporarily; the training head already computes with fp16 weights.
The tied identity, state_dict keys, trainable LoRA parameters, fixed task and
organizer files remain unchanged. Hooked, untied, trainable, inexact or unusual
modules keep their original path. A later unfreeze restores fp32 storage first.

For a 151936 x 1536 shared tensor, the proposed resident saving is 466747392
bytes (445.125 MiB), if the guard accepts it. This is a calculation, **not an
observed peak-memory or runtime improvement**. The temporary cast used by the
training head may also disappear, but its peak impact must be measured.

Local CPU validation: **15 passed, 5 CUDA cases skipped**, using PyTorch
2.11.0+cpu. Tests cover exact forward/input gradients both outside AMP and with
CPU bf16 AMP, embedding output dtype, tie/Parameter identity, fallback on
inexact/NaN/Inf or observed modules, later unfreeze and replaced weights.
The five CUDA cases cover fp16 AMP and the actual checked-copy kernel.
The full suite contains 158 cases with v17's decoder cases included.

GPU compilation, all five CUDA tests, full model behavior, quality, speed and
all ten contest checks are **pending**. v16 remains the latest fully verified
candidate. No 70–90% time reduction is claimed.

Implementation basis checked in the exact installed package sources:
`soup_cli.utils.gpu.resolve_frozen_base_load_dtype` selects fp16 on T4;
`peft.utils.other.prepare_model_for_kbit_training` subsequently upcasts ordinary
fp16 parameters. The runtime value proof is mandatory even if those upstream
policies change; it does not assume the conversion was lossless.

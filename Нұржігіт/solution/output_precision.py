"""Avoid widening the entire vocabulary output just to compute fp32 metrics.

Accelerate 1.14.0 utils.operations.ConvertOutputsToFp32 normally widens logits.
Our loss/entropy reductions already load each element into fp32. Keep the exact
fp16 logits while preserving the output container, loss and all other fields.
No weights, labels, training arguments or optimizer behavior are changed.
"""
import torch


def convert_outputs(output, original):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if (isinstance(output, CausalLMOutputWithPast)
            and isinstance(output.logits, torch.Tensor)
            and output.logits.is_cuda and output.logits.dtype == torch.float16
            and output.logits.ndim == 3 and output.logits.shape[-1] >= 1024):
        return type(output)(**{
            name: value if name == 'logits' else original(value)
            for name, value in output.items()
        })
    return original(output)


def patch_output_precision():
    from accelerate.utils import operations
    original = operations.convert_to_fp32
    if getattr(original, '_team_output_precision', False):
        return

    def convert_to_fp32(output):
        return convert_outputs(output, original)

    convert_to_fp32._team_output_precision = True
    operations.convert_to_fp32 = convert_to_fp32

"""Keep frozen Qwen MLP intermediates in the active fp16 compute dtype.

Reuses the attributed organizer SwiGLU; it is not an additional own kernel.
Input gradients accumulate in the original input dtype via separate casts.
"""
import torch
from .swiglu import swiglu


def patch_mlp_intermediates():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from bitsandbytes.nn import Linear4bit
    original = Qwen2MLP.forward
    if getattr(original, "_team_mlp_precision", False):
        return

    def forward(self, x):
        layers = (self.gate_proj, self.up_proj, self.down_proj)
        if (not x.is_cuda or x.dtype != torch.float32
                or not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.float16
                or getattr(self.config, "hidden_act", "silu") != "silu"
                or not all(isinstance(m, Linear4bit) and not m.weight.requires_grad
                           and m.bias is None for m in layers)):
            return original(self, x)
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        return self.down_proj(swiglu(gate, up)).to(x.dtype)

    forward._team_mlp_precision = True
    Qwen2MLP.forward = forward

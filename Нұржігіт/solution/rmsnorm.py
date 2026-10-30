"""Original fused implementation of Qwen2's frozen-weight RMSNorm.

Preserves Qwen2's normalization -> input-dtype cast -> weight multiply order,
including the corresponding cast in backward. Weight gradients use the original
implementation instead. Equation/source: Transformers 5.16.1 Qwen2RMSNorm.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def frozen_rms_forward(x, w, out, rstd, WIDTH: tl.constexpr,
                       EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(x + row * WIDTH + col, col < WIDTH, other=0.0).to(tl.float32)
    weight = tl.load(w + col, col < WIDTH, other=0.0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(value * value, 0) / WIDTH + EPS)
    normalized = (value * inverse).to(x.dtype.element_ty).to(tl.float32)
    tl.store(out + row * WIDTH + col, normalized * weight, col < WIDTH)
    tl.store(rstd + row, inverse)


@triton.jit
def frozen_rms_backward(x, w, grad, rstd, dx,
                        WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(x + row * WIDTH + col, col < WIDTH, other=0.0).to(tl.float32)
    weight = tl.load(w + col, col < WIDTH, other=0.0).to(tl.float32)
    upstream = tl.load(grad + row * WIDTH + col, col < WIDTH, other=0.0).to(tl.float32)
    g = (upstream * weight).to(x.dtype.element_ty).to(tl.float32)
    inverse = tl.load(rstd + row)
    dot = tl.sum(g * value, 0) / WIDTH
    result = inverse * g - value * inverse * inverse * inverse * dot
    tl.store(dx + row * WIDTH + col, result, col < WIDTH)


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        width = x.shape[-1]
        out = torch.empty(x.shape, device=x.device,
                          dtype=torch.promote_types(x.dtype, weight.dtype))
        inv = torch.empty(x.numel() // width, dtype=torch.float32, device=x.device)
        frozen_rms_forward[(inv.numel(),)](
            x, weight, out, inv, width, eps, triton.next_power_of_2(width),
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, inv)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        dx = torch.empty_like(x)
        width = x.shape[-1]
        frozen_rms_backward[(inv.numel(),)](
            x, weight, grad.contiguous(), inv, dx, width,
            triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
        )
        return dx, None, None


def frozen_rmsnorm(x, weight, eps=1e-6):
    return _FrozenRMSNorm.apply(x, weight, eps)


def patch_frozen_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_team_norm", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if (not hidden_states.is_cuda or not hidden_states.is_contiguous()
                or self.weight.requires_grad or not self.weight.is_contiguous()
                or hidden_states.dtype not in (torch.float16, torch.float32)
                or self.weight.dtype not in (torch.float16, torch.float32)
                or hidden_states.shape[-1] > 8192 or hidden_states.numel() == 0):
            return original(self, hidden_states)
        return frozen_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_team_norm = True
    cls.forward = forward

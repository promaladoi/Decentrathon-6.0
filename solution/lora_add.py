"""Original fused LoRA scale/add; semantics from PEFT 0.20.0 Linear4bit.

Writes a new output, so the base result does not need a defensive clone.
Dropout and both LoRA matrix multiplications remain the original modules.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def lora_scale_add(base, update, out, N: tl.constexpr, SCALE: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(base + i, i < N, other=0).to(tl.float32)
    u = tl.load(update + i, i < N, other=0).to(tl.float32)
    scaled = (u * SCALE).to(update.dtype.element_ty).to(tl.float32)
    tl.store(out + i, b + scaled, i < N)


@triton.jit
def lora_update_gradient(grad, du, N: tl.constexpr, SCALE: tl.constexpr,
                         BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(grad + i, i < N, other=0).to(du.dtype.element_ty).to(tl.float32)
    tl.store(du + i, g * SCALE, i < N)


class _ScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, update, scale):
        ctx.update_dtype, ctx.scale = update.dtype, scale
        out = torch.empty_like(base)
        lora_scale_add[(triton.cdiv(base.numel(), 1024),)](
            base, update, out, base.numel(), scale, 1024, enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        du = torch.empty(grad.shape, device=grad.device, dtype=ctx.update_dtype)
        lora_update_gradient[(triton.cdiv(grad.numel(), 1024),)](
            grad, du, grad.numel(), ctx.scale, 1024, enable_fp_fusion=False)
        return grad, du, None


def scale_add(base, update, scale):
    return _ScaleAdd.apply(base, update, scale)


def patch_lora_add():
    from peft.tuners.lora.bnb import Linear4bit
    original = Linear4bit.forward
    if getattr(original, '_team_scale_add', False):
        return

    def forward(self, x, *args, **kwargs):
        active = self.active_adapters
        if (args or kwargs or not x.is_cuda or not torch.is_autocast_enabled('cuda')
                or self.disable_adapters or self.merged or len(active) != 1
                or active[0] not in self.lora_A or active[0] in self.lora_variant):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        a = active[0]
        base = self.base_layer(x)
        update = self.lora_B[a](self.lora_A[a](self.lora_dropout[a](x)))
        scale = self.scaling[a]
        if (base.dtype == torch.float32 and update.dtype in (torch.float16, torch.float32)
                and base.shape == update.shape and base.is_contiguous()
                and update.is_contiguous() and scale in (0.5, 1.0, 2.0)):
            return scale_add(base, update, scale)
        return base + update * scale

    forward._team_scale_add = True
    Linear4bit.forward = forward

"""Reuse a frozen head's fp16 cast and omit exactly zero gradient rows.

Forward still emits every token's logits. Only rows whose entire upstream
gradient is zero skip the vocabulary backward GEMM; their input gradient is
exactly zero. The cast cache is built inside each run and invalidated by normal
parameter mutation/replacement. It is not a prebuilt/compiler/model cache.
"""
from functools import wraps
import math
import types

import torch
import triton
import triton.language as tl


@triton.jit
def head_active_tiles(grad, flags, V: tl.constexpr, PARTS: tl.constexpr,
                      STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(grad + row * STRIDE + col, col < V, other=0.0)
    # NaNs compare nonzero and must never be silently discarded.
    any_nonzero = tl.max((value != 0).to(tl.int32), 0)
    tl.store(flags + row * PARTS + part, any_nonzero)


class _FrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, half_weight):
        ctx.save_for_backward(half_weight)
        ctx.input_shape, ctx.input_dtype = x.shape, x.dtype
        return torch.nn.functional.linear(x.to(half_weight.dtype), half_weight)

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        g = grad.reshape(-1, weight.shape[0]).contiguous()
        rows, vocab = g.shape
        parts = triton.cdiv(vocab, 4096)
        flags = torch.empty((rows, parts), device=g.device, dtype=torch.int32)
        head_active_tiles[(rows, parts)](g, flags, vocab, parts, g.stride(0), 4096, num_warps=4)
        indices = flags.any(dim=1).nonzero(as_tuple=True)[0]
        if indices.numel() == rows:
            dx = torch.matmul(g, weight)
        else:
            dx = torch.zeros((rows, weight.shape[1]), device=g.device, dtype=g.dtype)
            if indices.numel():
                selected = g.index_select(0, indices)
                values = torch.matmul(selected, weight)
                dx.index_copy_(0, indices, values)
        return dx.reshape(ctx.input_shape).to(ctx.input_dtype), None


def frozen_head(x, weight):
    return _FrozenHead.apply(x, weight)


def install_head(head):
    if type(head) is not torch.nn.Linear or getattr(head, '_team_head_installed', False):
        return
    original = head.forward

    def forward(self, x):
        w = self.weight
        if (not x.is_cuda or not w.is_cuda or w.requires_grad or self.bias is not None
                or w.dtype != torch.float32 or x.dtype not in (torch.float16, torch.float32)
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.float16
                or self.out_features < 1024 or x.numel() == 0):
            self.__dict__.pop('_team_half_cache', None)
            return original(x)
        key = (w._version, w.data_ptr(), w.device, tuple(w.shape))
        cached = getattr(self, '_team_half_cache', None)
        if cached is None or cached[0] is not w or cached[1] != key:
            with torch.inference_mode(False), torch.no_grad():
                half = w.to(torch.float16)
            cached = (w, key, half)
            self._team_half_cache = cached
        return frozen_head(x, cached[2])

    head.forward = types.MethodType(forward, head)
    head._team_head_installed = True


def patch_frozen_head():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    original = Qwen2ForCausalLM.forward
    if getattr(original, '_team_frozen_head', False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        install_head(self.lm_head)
        return original(self, *args, **kwargs)

    forward._team_frozen_head = True
    Qwen2ForCausalLM.forward = forward

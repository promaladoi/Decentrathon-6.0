"""Isolated native-PyTorch attention alternatives, not installed in training.

All outputs contain the real prefix tokens in batch order, [tokens, H*D].
The dense baseline uses the teammate's current mask/GQA policy. Alternatives
keep explicit equal head counts so T4 can use native efficient attention.
Packed attention calls the pinned torch2.11 Cutlass operator with int32 sequence
boundaries; its registered native autograd receives BOTH explicit max lengths.
No external attention package, custom kernel, cache or RNG operation is used.

Sources: PyTorch v2.11.0 attention.cpp, cuda/attention.cu,
cuda/attention_backward.cu, cuda/sdp_utils.cpp and derivatives.yaml.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Layout:
    lengths: tuple
    sequence: int
    total: int
    maximum: int
    all_full: bool
    cumulative: torch.Tensor
    dense_mask: object
    padding_bias: torch.Tensor


def make_layout(lengths, sequence, device):
    lengths = tuple(int(n) for n in lengths)
    if not lengths or any(n <= 0 or n > sequence for n in lengths):
        raise ValueError('only nonempty right-padded prefix rows are supported')
    cumulative = [0]
    for n in lengths:
        cumulative.append(cumulative[-1] + n)
    cu = torch.tensor(cumulative, device=device, dtype=torch.int32)
    positions = torch.arange(sequence, device=device)
    counts = torch.tensor(lengths, device=device)
    keys = positions[None, :] < counts[:, None]
    causal = positions[:, None] >= positions[None, :]
    all_full = all(n == sequence for n in lengths)
    dense_mask = None if all_full else (causal[None, None] & keys[:, None, None, :])
    # This bias is padding-only; the efficient kernel applies causal structure.
    padding_bias = keys[:, None, None, :]
    return Layout(lengths, sequence, cumulative[-1], max(lengths), all_full,
                  cu, dense_mask, padding_bias)


def _validate(q, k, v, layout, equal_heads=False):
    if (not isinstance(layout, Layout) or q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or not all(t.is_cuda and t.dtype == torch.float16 for t in (q, k, v))
            or not (q.device == k.device == v.device == layout.cumulative.device)
            or q.shape[0] != len(layout.lengths) or k.shape[0] != q.shape[0]
            or v.shape != k.shape or q.shape[2] != layout.sequence
            or k.shape[2] != q.shape[2] or q.shape[-1] != k.shape[-1]
            or any(t.stride(-1) != 1 for t in (q, k, v))
            or (equal_heads and q.shape[1] != k.shape[1])):
        raise ValueError('unsupported CUDA fp16 padded attention inputs')
    if q.shape[1] % k.shape[1]:
        raise ValueError('query heads must be divisible by key heads')


def _context(force_efficient):
    if not force_efficient:
        return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)


def flatten_real(padded, layout):
    return torch.cat([padded[b, :, :n, :].transpose(0, 1)
                      for b, n in enumerate(layout.lengths)], dim=0).flatten(1)


def current_padded(q, k, v, layout, scale=None, force_efficient=False):
    """Current teammate SDPA policy; k/v may have HK heads when no padding."""
    _validate(q, k, v, layout)
    grouped = q.shape[1] != k.shape[1]
    if grouped and not layout.all_full:
        raise ValueError('current padded policy requires explicit repeated heads')
    kwargs = {'enable_gqa': True} if grouped else {}
    with _context(force_efficient):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=layout.dense_mask,
            dropout_p=0.0, is_causal=layout.all_full and layout.sequence > 1,
            scale=scale, **kwargs)
    return flatten_real(out, layout)


def equalheads_padded(q, k, v, layout, scale=None, force_efficient=False):
    _validate(q, k, v, layout, equal_heads=True)
    return current_padded(q, k, v, layout, scale, force_efficient)


def per_row_causal(q, k, v, layout, scale=None, force_efficient=False):
    """One native causal/no-mask SDPA per row; K/V already explicitly repeated."""
    _validate(q, k, v, layout, equal_heads=True)
    outputs = []
    with _context(force_efficient):
        for b, n in enumerate(layout.lengths):
            out = F.scaled_dot_product_attention(q[b:b+1, :, :n, :],
                k[b:b+1, :, :n, :], v[b:b+1, :, :n, :], attn_mask=None,
                dropout_p=0.0, is_causal=n > 1, scale=scale, enable_gqa=False)
            outputs.append(out[0].transpose(0, 1))
    return torch.cat(outputs, dim=0).flatten(1)


def causal_padding_bias(q, k, v, layout, scale=None, force_efficient=False):
    """Pinned native efficient SDPA: separate causal flag and key padding bias.

    The math backend rejects simultaneous mask+causal, so this diagnostic fails
    closed unless native efficient attention is available for these parameters.
    It is not a general public replacement for SDPA.
    """
    _validate(q, k, v, layout, equal_heads=True)
    params = torch.backends.cuda.SDPAParams(q, k, v,
        None if layout.all_full else layout.padding_bias,
        0.0, layout.sequence > 1, False)
    if not torch.backends.cuda.can_use_efficient_attention(params, debug=False):
        raise RuntimeError('native efficient SDPA required for causal+padding bias')
    with _context(True):
        out = F.scaled_dot_product_attention(q, k, v,
            attn_mask=None if layout.all_full else layout.padding_bias,
            dropout_p=0.0, is_causal=layout.sequence > 1, scale=scale,
            enable_gqa=False)
    return flatten_real(out, layout)


def packed_tensors(q, k, v, cumulative, maximum, scale=None):
    """Native efficient attention on [1,total_tokens,H,D], with known row bounds.

    Caller constructs contiguous CUDA int32 cumulative lengths. Both explicit
    max lengths are essential: pinned CUDA's internal max-K variable is zero in
    varlen mode, but the return metadata uses the supplied max-K argument.
    """
    if (not torch.__version__.split('+')[0].startswith('2.11.')
            or q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or k.shape != v.shape
            or not all(t.is_cuda and t.dtype == torch.float16 for t in (q, k, v))
            or not (q.device == k.device == v.device == cumulative.device)
            or cumulative.dtype != torch.int32 or cumulative.ndim != 1
            or not cumulative.is_contiguous() or maximum < 1 or maximum > q.shape[1]
            or q.shape[-1] % 8 or any(t.stride(-1) != 1 for t in (q, k, v))):
        raise ValueError('unsupported pinned native packed attention inputs')
    result = torch.ops.aten._efficient_attention_forward.default(
        q, k, v, None, cumulative, cumulative, maximum, maximum,
        0.0, 1, True, scale=scale)
    return result[0]


def packed_from_padded(q, k, v, layout, scale=None, force_efficient=False):
    """Diagnostic includes three native gathers from existing padded inputs."""
    _validate(q, k, v, layout, equal_heads=True)
    packed = [torch.cat([t[b, :, :n, :].transpose(0, 1)
                        for b, n in enumerate(layout.lengths)], dim=0).unsqueeze(0)
              for t in (q, k, v)]
    return packed_tensors(*packed, layout.cumulative, layout.maximum, scale).squeeze(0).flatten(1)


VARIANTS = {'current': current_padded, 'equalheads': equalheads_padded,
            'rows': per_row_causal, 'packed': packed_from_padded,
            'causal_bias': causal_padding_bias}

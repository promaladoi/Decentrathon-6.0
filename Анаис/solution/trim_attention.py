"""Exact SDPA on each row's occupied span; padding cannot attend to real tokens.

Only queries whose outputs are unused are omitted. The caller keeps LoRA dropout
on the original padded batch, so this does not change random draws or batch order.
"""
import torch
import torch.nn.functional as F


def occupied_spans(attention_mask):
    mask = attention_mask.to(torch.int32)
    first = mask.argmax(dim=1)
    end = mask.shape[1] - mask.flip((1,)).argmax(dim=1)
    return torch.stack((first, end), dim=1).tolist()


def trimmed_sdpa(q, k, v, mask, spans, *, scale, is_causal=False, enable_gqa=False):
    batch, _, length, _ = q.shape
    kwargs = dict(attn_mask=mask, dropout_p=0.0, scale=scale,
                  is_causal=is_causal, enable_gqa=enable_gqa)
    # Avoid four smaller launches when there is little padding to remove.
    if sum((end-start)**2 for start, end in spans) >= .75*batch*length*length:
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    output = torch.zeros_like(q)
    for b, (start, end) in enumerate(spans):
        if end <= start:
            continue
        row_mask = mask
        if mask is not None:
            if mask.ndim == 4 and mask.shape[0] == batch:
                row_mask = mask[b:b+1]
            query_slice = slice(None) if row_mask.shape[-2] == 1 else slice(start, end)
            row_mask = row_mask[..., query_slice, start:end]
        kwargs['attn_mask'] = row_mask
        output[b:b+1, :, start:end] = F.scaled_dot_product_attention(
            q[b:b+1, :, start:end], k[b:b+1, :, start:end],
            v[b:b+1, :, start:end], **kwargs)
    return output

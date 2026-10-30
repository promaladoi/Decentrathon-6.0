"""Remove attention padding using PyTorch's own variable-length efficient SDPA.

The logical sequences and original positions remain separate through cumulative
lengths. RoPE/NF4/LoRA still use the shared exact path; LoRA dropout retains its
original full padded draws. No external attention kernel package is imported.
"""
import torch


def make_plan(rows, batch, seq_len):
    if rows.numel() == 0 or batch < 1 or seq_len < 1:
        return None
    # nonzero() in the shared forward emits rows in batch/position order.
    lengths = torch.bincount(rows // seq_len, minlength=batch)
    counts = lengths.cpu().tolist()
    if len(counts) != batch or min(counts) <= 0 or max(counts) > seq_len:
        return None
    cu = torch.cat([lengths.new_zeros(1), lengths]).cumsum(0).to(torch.int32)
    return {'rows': torch.arange(rows.numel(), device=rows.device, dtype=torch.int64),
            'cu': cu, 'max_len': max(counts), 'n': rows.numel()}


def attention(q, k, v, plan, scale):
    """The same native operator used by torch.nested jagged SDPA in Torch 2.11.

    Inputs are [1, total_tokens, heads, dim], with independently delimited batch
    sequences. The operator has its native registered autograd implementation.
    """
    out, _lse, _seed, _offset, _mq, _mk = torch.ops.aten._efficient_attention_forward(
        q, k, v, None, plan['cu'], plan['cu'], plan['max_len'], plan['max_len'],
        0.0, 1, True, scale=scale)
    return out


def can_skip_dense_mask(step,meta):
    """Only when every layer uses independently delimited causal sequences."""
    return (getattr(step,'_varlen_plan',None) is not None and meta['dim']<=128 and
            all(getattr(layer.self_attn,'is_causal',True) for layer,_ in meta['layers']))


def make_batch_init(original):
    def init(self, rows, batch, seq_len):
        original(self, rows, batch, seq_len)
        self._varlen_plan = make_plan(rows, batch, seq_len)
    init._varlen_attention = True
    return init


def make_attention(original):
    def forward(layer, projections, x, cos, sin, mask, step, meta):
        plan = getattr(step, '_varlen_plan', None)
        if (plan is None or meta['dim'] > 128 or
                not getattr(layer.self_attn, 'is_causal', True)):
            return original(layer, projections, x, cos, sin, mask, step, meta)
        from .team_vladimir import unpad
        q_parts, k_parts, v_parts, o_parts = projections
        q, k, v = unpad.QKV4bit.apply(x, q_parts[0], k_parts[0], v_parts[0], torch.float16)
        # Preserve the original q/k/v LoRA order and padded dropout RNG draws.
        lq = unpad._lora_out(q_parts, x, step)
        lk = unpad._lora_out(k_parts, x, step)
        lv = unpad._lora_out(v_parts, x, step)
        step.done_with(x)
        # The existing RoPE kernel accepts arbitrary output strides. Compact
        # storage uses all token rows; cos/sin retain their original positions.
        qp, kp, vp = unpad.RoPEPad.apply(
            q, lq, k, lk, v, lv, cos, sin, plan['rows'], 1, plan['n'],
            meta['hq'], meta['hk'], meta['dim'], meta['rep'])
        out = attention(qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2),
                        plan, layer.self_attn.scaling)
        a = out.squeeze(0).reshape(plan['n'], meta['hq']*meta['dim'])
        o = unpad.MatMulNF4.apply(a, o_parts[0].weight, None, o_parts[0].weight.quant_state)
        lo = unpad._lora_out(o_parts, a, step)
        step.done_with(a)
        return o if lo is None else o + lo
    forward._varlen_attention = True
    forward._can_skip_dense_mask = can_skip_dense_mask
    return forward


def patch_varlen_attention():
    # The internal schema and SM75 backend were inspected at this pinned version.
    if not torch.__version__.startswith('2.11.') or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (7, 5):
        return False
    from .team_vladimir import unpad
    if getattr(unpad._attention, '_varlen_attention', False):
        return True
    unpad._Batch.__init__ = make_batch_init(unpad._Batch.__init__)
    unpad._attention = make_attention(unpad._attention)
    return True

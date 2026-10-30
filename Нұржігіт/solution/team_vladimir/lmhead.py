"""lm_head, cross-entropy and the per-token metrics in row chunks, without the full logits.

In the base run the 4 x 512 x 151,936 logits exist several times at once: lm_head writes them in
fp16 (0.6 GB), the loss upcasts a fp32 copy (1.2 GB) and keeps its log-softmax (1.2 GB) for
backward, accelerate converts the output logits to fp32 (1.2 GB), TRL's token accuracy makes one
more contiguous copy, and TRL's entropy and Soup's per-row rewind losses run softmaxes over them.
That is the largest block of the training step's peak memory and about 7% of its time.

Here the logits of 512 tokens at a time are multiplied out (fp16, cuBLAS, the same product
lm_head computes under autocast) and one Triton program per token reads its row of 151,936 logits:
an online softmax (Milakov and Gimelshein, https://arxiv.org/abs/1805.02867) gives the
log-sum-exp, the entropy and the arg-max in one pass, and for tokens with a label a second pass
writes the gradient of the cross-entropy, softmax - one_hot, over the logits in place. One more
matmul turns that into the gradient of the hidden state right away, so nothing of the vocabulary
size is kept for backward: backward only scales that gradient by the loss scale. Tokens without
a label (prompt tokens) only need their entropy, so they skip the gradient.

The loss is the same sum of per-token cross-entropies over the same tokens divided by the same
num_items_in_batch as transformers' ForCausalLMLoss. TRL's entropy and token accuracy and Soup's
per-row rewind losses are computed from these per-token values by the same formulas, through
LazyLogits, which stands in for the logits tensor where they read it (see the patches below).
Code written by us.
"""

import torch
import triton
import triton.language as tl

CHUNK = 512  # tokens per logits chunk: 512 x 151,936 fp16 = 156 MB
BLOCK_V = 4096
GRAD_SCALE = 1024.0  # keeps small probabilities out of fp16 subnormals; divided out in backward


@triton.jit
def lm_ce_kernel(LOGITS, TARGET, ENT, AMAX, CE, stride, V, grad_scale, BLOCK_V: tl.constexpr,
                 WITH_GRAD: tl.constexpr):
    row = tl.program_id(0)
    ptr = LOGITS + row.to(tl.int64) * stride
    target = -1
    if WITH_GRAD:
        target = tl.load(TARGET + row).to(tl.int32)
    m = -1e30  # running max
    s = 0.0  # sum of exp(z - m)
    u = 0.0  # sum of exp(z - m) * (z - m), for the entropy
    best = -1e30
    best_i = 0
    zt = 0.0  # the target's logit
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.where(mask, z, -1e30)
        bmax = tl.max(z, axis=0)
        bi = tl.argmax(z, axis=0)  # first index of the block's max, as torch.argmax
        best_i = tl.where(bmax > best, start + bi, best_i)
        best = tl.maximum(best, bmax)
        new_m = tl.maximum(m, bmax)
        alpha = tl.exp(m - new_m)
        e = tl.exp(z - new_m)
        u = alpha * (u + s * (m - new_m)) + tl.sum(e * (z - new_m), axis=0)
        s = alpha * s + tl.sum(e, axis=0)
        m = new_m
        if WITH_GRAD:
            zt += tl.sum(tl.where(cols == target, z, 0.0), axis=0)
    log_s = tl.log(s)
    tl.store(ENT + row, log_s - u / s)  # -sum(p * log p)
    tl.store(AMAX + row, best_i.to(tl.int64))
    if WITH_GRAD:
        lse = m + log_s
        tl.store(CE + row, lse - zt)
        for start in range(0, V, BLOCK_V):  # d(CE)/dz = softmax - one_hot, in place
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            z = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(z - lse)
            g = tl.where(cols == target, p - 1.0, p) * grad_scale
            tl.store(ptr + cols, g.to(LOGITS.dtype.element_ty), mask=mask)


def _stats(x16, w16, idx, targets, ent, amax, ce, grad_x):
    """Fill the statistics (and, with targets, the hidden-state gradient) of tokens `idx`."""
    with_grad = targets is not None
    vocab = w16.shape[0]
    if idx.numel() == 0:
        return
    for chunk in idx.split(CHUNK):
        n = chunk.numel()
        logits = torch.mm(x16.index_select(0, chunk), w16.t())
        tgt = targets.index_select(0, chunk) if with_grad else chunk
        e, a = torch.empty(n, device=x16.device), torch.empty(n, device=x16.device,
                                                               dtype=torch.int64)
        c = torch.empty(n, device=x16.device)
        lm_ce_kernel[(n,)](logits, tgt, e, a, c, logits.stride(0), vocab, GRAD_SCALE,
                           BLOCK_V=BLOCK_V, WITH_GRAD=with_grad, num_warps=8)
        ent.index_copy_(0, chunk, e)
        amax.index_copy_(0, chunk, a)
        if with_grad:
            ce.index_copy_(0, chunk, c)
            grad_x.index_copy_(0, chunk, torch.mm(logits, w16).float())
        del logits


class LMHeadLoss(torch.autograd.Function):
    """sum over labelled tokens of CE(x @ W.T, target) / denom; fills `stats` with the per-token
    entropy, arg-max and cross-entropy."""

    @staticmethod
    def forward(ctx, x, weight, targets, denom, stats):
        tokens, hidden = x.shape
        x16 = x.to(torch.float16)
        w16 = weight.to(torch.float16)  # what autocast feeds lm_head
        ent = torch.empty(tokens, device=x.device)
        amax = torch.empty(tokens, device=x.device, dtype=torch.int64)
        ce = torch.zeros(tokens, device=x.device)
        grad_x = torch.zeros(tokens, hidden, device=x.device)
        valid = targets != -100
        _stats(x16, w16, valid.nonzero().squeeze(1), targets, ent, amax, ce, grad_x)
        _stats(x16, w16, (~valid).nonzero().squeeze(1), None, ent, amax, ce, grad_x)
        del w16, x16
        stats.update(entropy=ent, argmax=amax, ce=ce, valid=valid)
        ctx.save_for_backward(grad_x, denom)
        ctx.x_dtype = x.dtype
        return ce.sum() / denom

    @staticmethod
    def backward(ctx, grad_loss):
        grad_x, denom = ctx.saved_tensors
        scale = grad_loss.float() / (denom.float() * GRAD_SCALE)
        return (grad_x * scale).to(ctx.x_dtype), None, None, None, None


def lm_head_loss(x, weight, targets, num_items_in_batch):
    """ForCausalLMLoss: reduction "sum" over num_items_in_batch when given, else the mean."""
    if num_items_in_batch is None:
        denom = (targets != -100).sum().float()
    elif torch.is_tensor(num_items_in_batch):
        denom = num_items_in_batch.to(x.device)
    else:
        denom = torch.tensor(num_items_in_batch, device=x.device)
    stats = {}
    loss = LMHeadLoss.apply(x, weight, targets, denom, stats)
    return loss, stats


class LazyLogits:
    """Stands in for the (batch, seq_len, vocab) logits in the model output.

    The full logits never exist in this run. The three readers of `outputs.logits` in a training
    step get the per-token values they compute, through the patches below:
    TRL's entropy_from_logits (entropy per position), TRL's token accuracy
    (`logits[..., :-1, :].contiguous().argmax(dim=-1)`) and Soup's rewind row_losses. Anything
    else raises, so a new reader cannot silently get wrong values.
    """

    def __init__(self, batch, seq_len, vocab, rows, stats):
        self.shape = torch.Size((batch, seq_len, vocab))
        self._rows, self._stats = rows, stats

    def _full(self, values, fill):
        batch, seq_len, _ = self.shape
        out = values.new_full((batch * seq_len,), fill)
        out.index_copy_(0, self._rows, values)
        return out.view(batch, seq_len)

    def entropy(self):
        return self._full(self._stats["entropy"], 0.0)

    def predictions(self):
        return self._full(self._stats["argmax"], -1)

    def token_ce(self):
        """Cross-entropy of position t predicting label t + 1, 0 where that label is -100."""
        return self._full(self._stats["ce"], 0.0)

    def __getitem__(self, key):
        if key == (Ellipsis, slice(None, -1, None), slice(None, None, None)):
            return _ShiftedLazyLogits(self)
        raise TypeError(f"LazyLogits does not support indexing with {key!r}")

    def __getattr__(self, name):
        raise AttributeError(f"LazyLogits has no {name!r}: the full logits are not materialized")


class _ShiftedLazyLogits:
    """`logits[..., :-1, :]`, as TRL's token accuracy reads it."""

    def __init__(self, parent):
        self._parent = parent

    def contiguous(self):
        return self

    def argmax(self, dim=-1):
        if dim not in (-1, 2):
            raise TypeError("only the argmax over the vocabulary is available")
        return self._parent.predictions()[:, :-1]

    def __getattr__(self, name):
        raise AttributeError(f"shifted LazyLogits has no {name!r}")


def make_entropy_from_logits(original):
    def entropy_from_logits(logits, chunk_size=128):
        if isinstance(logits, LazyLogits):
            return logits.entropy()
        return original(logits, chunk_size)

    return entropy_from_logits


def make_row_losses(original):
    def row_losses(logits, labels):
        if not isinstance(logits, LazyLogits):
            return original(logits, labels)
        # Soup's row_losses: per row, the mean over labels[:, 1:] != -100 of the token CE
        with torch.no_grad():
            mask = labels[:, 1:] != -100
            token_ce = logits.token_ce()[:, :-1].to(labels.device)
            tokens = mask.sum(dim=1)
            totals = (token_ce * mask.to(token_ce.dtype)).sum(dim=1)
            means = totals / tokens.clamp(min=1).to(totals.dtype)
        return [float(v) for v in means], [int(v) for v in tokens]

    return row_losses


def patch_metric_readers():
    """Point TRL's entropy and Soup's rewind row losses at LazyLogits. Done when the fast forward
    first runs, from the modules already loaded: nothing is imported early, so Soup's tests that
    swap in fake modules are unaffected."""
    import sys

    sft = sys.modules.get("trl.trainer.sft_trainer")
    if sft is not None and not getattr(sft.entropy_from_logits, "_solution", False):
        sft.entropy_from_logits = make_entropy_from_logits(sft.entropy_from_logits)
        sft.entropy_from_logits._solution = True
    rewind = sys.modules.get("soup_cli.trainer.rewind_hf")
    if rewind is not None and not getattr(rewind.row_losses, "_solution", False):
        rewind.row_losses = make_row_losses(rewind.row_losses)
        rewind.row_losses._solution = True


def warmup(vocab=151936):
    """Compile the two variants of lm_ce_kernel; see warmup.py."""
    f16, f32, i64 = torch.float16, torch.float32, torch.int64
    for with_grad in (True, False):
        lm_ce_kernel.warmup(f16, i64, f32, i64, f32, vocab, vocab, GRAD_SCALE, grid=(1,),
                            BLOCK_V=BLOCK_V, WITH_GRAD=with_grad, num_warps=8)

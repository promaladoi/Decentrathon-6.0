# Combined v13 experiment

Vladimir's team contribution is imported from collaboration repository commit
`6436ef5683892692b9e41f2c5fe6bb563107a128`, directory `Владимир/solution`, into `solution/team_vladimir/`.
The implementation is credited to Vladimir and his development tools, not claimed
as independently written by Nurzhigit/Codex. Only test imports are relocated.
Organizer example SwiGLU kernel names remain unchanged and excluded from own-kernel share.

Nurzhigit/Codex adds `solution/parallel_head.py`: a parallel tiled vocabulary
reduction that replaces the serial online vocabulary loop in the shared head.
The same logits, entropy, predictions, token losses and hidden-state gradient
are computed for the same real tokens. All existing fallbacks remain.
No harness, fixed task, model/data, dropout, optimizer or Soup switch changes.

The v13 version must pass GPU unit tests, regression tests and a fresh paired
benchmark before its result is considered validated. Older files remain for
the v1-v12 experiment history; apply() selects only the shared path and this patch.

The v14 experiment additionally keeps PyTorch's full-shape native dropout mask
and RNG consumption, but applies that mask and its backward only on real-token
rows. `solution/compact_dropout.py` and its GPU tests verify exact output,
input gradient and RNG state equality against the original padded dropout.

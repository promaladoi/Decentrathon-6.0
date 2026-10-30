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

v14's direct native_dropout mask source failed exact equivalence tests on T4.
Repair 3016d55 uses the same public functional.dropout operator as the original.
v15 ef10dd1 registers current kernels before the existing shared compile thread
starts, so compilation overlaps model startup. No prebuilt cache or harness
timing change is introduced. Both revisions still need GPU validation.

Correction after v15 failure: the RNG-mapping diagnosis above was not established.
The compact kernel assumed contiguous row-index storage; tests passed stride-2/3
views. v16 normalizes the indices and retains strict output/gradient/RNG tests.


v21 additionally integrates Vladimir's startup revision `987da36e1efa90652e1a7f8c48afef50685dbc40` for warmup.py,
rmsnorm.py, mlp.py and unpad.py. Its changes reduce compiled variants, compile
in first-use order and join the in-process compilation thread before step 1.
The initial RMSNorm reuses the same residual-add kernel with an exact zero
residual; head gather/scatter share a runtime-direction variant. Existing
fallbacks and original organizer SwiGLU exclusions remain. This is credited to
Vladimir, not claimed as independent Nurzhigit/Codex work. No precompiled cache
is imported and all work remains inside the timed process.

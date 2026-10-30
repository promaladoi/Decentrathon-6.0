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

v26 adapts Vladimir's `13d377043fa4cdad121973a3f6179d546918e9f4` launcher preparation
idea through `launcher_warmup.py`, including all already-loaded shared and
Nurzhigit kernels, with function/kernel alias deduplication. Attribution remains
with Vladimir for this startup optimization; no timeline module is imported.

v36 imports Merei's `backward_sums.py`, `batch_metadata.py` and associated tests
from exact source `29419f43d9b9173d1ba5d6824a31589ad3b84df7` under
`solution/team_merei/`. Their authorship remains with Merei and his tools. We
adapt warmup/package imports and integrate the metadata with our native varlen
plan. Ordered FP32 sums use the original native GEMMs; no new dot algorithm is
enabled. The imported strict tests remain in the production validation suite.

v43 integrates Merei's unchanged `package_metadata_snapshot.py` and
`test_package_metadata_snapshot.py` from collaboration source
`6a4f577e95605547b1b713013db1ffc44aa620bd:Мерей/solution/`. Authorship stays
with Merei and his tools. Their copy SHA256 values are respectively
`f9fb19d5577f8276aaebaf1df16ecb867f791575b685fdfefddb1b84659ce3d3`
and `25f4d0892c34d2341e49ae30c4f781ae3dc55a3756136e53e9e7cd1b797276c2`.
Nurzhigit adds the deferred native-import integration and its independent
contracts/proof. This CPU helper is never counted as an own Triton kernel.
It is candidate work pending the combined T4 benchmark, not a transfer of
Merei's earlier performance claims to this source revision.


## v44: combined team memory candidate (pending T4)

The active experimental path now imports Merei's complete corrected stage2
Python solution and all tests from `6a4f577e95605547b1b713013db1ffc44aa620bd`.
Merei and the underlying Vladimir contributions retain their authorship.
Nurzhigit's additions are exact tied frozen storage, its integration tests and
the validation runner. The 42 unchanged imported files have provenance hashes.
The package initializer is adapted; no performance result is inherited.
See `CODEX_V44_EXPERIMENT.md`. Older sections describe historical revisions,
not the current experiment's imports. The public selected solution stays v34.

# v28: faster complete tokenizer serialization for internal dataset fingerprints

Experimental on top of v27. The separate CPU profile spent 7.55 seconds in
datasets' Python dill tokenizer reducer (profiled time, not contest time).
A real Qwen tokenizer on the local CPU took 1.667 s with datasets dill and
0.248 s with C pickle before this optimization; these are diagnostics only.

A source-guarded rewrite retains datasets' exact cache/warning exclusions and
backend truncation/padding normalization/restoration. Only its final reduction
of the full Qwen2Tokenizer state uses C pickle. Dictionaries are canonicalized;
all vocabulary, merges, chat template, settings and special-token state remain
in the serialization. Other classes and unknown reducer versions fall back.

This changes internal cache-key bytes, never training inputs. No tokenized data
is reused across runs, no preprocessing is moved outside the timer, and the
original formatting, target validation and mapping still execute. Tests cover
roundtrip tokenization, hashing stability, relevant-state invalidation and map
outputs; the unchanged full-task/rewind/loss/quality rules must still all pass.

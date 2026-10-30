# v49: compact the proven frozen tied pair immediately after trainer construction

Candidate on v48, whose math is v45. Move the existing exact FP32-to-FP16 storage
proof earlier: after the native Rewind trainer factory returns and its private
wrapper qualifies. This avoids retaining the redundant FP32 storage through
later trainer setup. The original first-forward hook remains a fallback.

No values, quantization, optimizer, trainable parameter identities, module output
dtypes, RNG or kernels change. The same all-or-nothing exact-value check and
embedding/head wrappers run in the same timed process. Unknown models, observed
modules, untied/trainable/inexact weights keep the existing safe behavior.

Tests cover value identity, tied Parameter identity, idempotence, FP32 public
embedding/linear outputs and observer fallback. Full paired measurements and
original ten checks are still required; no time or peak saving is claimed yet.
Verified v45 remains selected while this experiment is pending.

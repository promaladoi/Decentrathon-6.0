# v51: build causal target partitions with batch metadata

Candidate based on the exact selected v45 source, not the slower v49 early-storage variant.
The original task, harness and evaluation files are unchanged.

The existing batch-metadata Triton launch also builds shifted causal targets and
the supervised/prompt index lists in stable order. Its existing small host read
now includes the supervised count. The private head reuses these lists instead
of synchronizing two `nonzero` operations after the model forward.
No floating-point formula, GEMM shape, token order, dropout or RNG call changes.
General/public head calls and larger metadata layouts retain the existing path.

Four bounded index/target buffers use at most 64 KiB together. Whether removing
the late synchronizations improves whole-command time or peak memory must be
measured; no gain, loss value, 10/10 result, or <=27-second result is claimed yet.

Before training: exact causal-target and stable-partition tests including
strided labels, padding boundaries, all ignored targets and matching private
AMP scales; then the full solution/native-parity/Soup suites. A complete run
must retain both original base/solution repeats and all ten organizer checks.
User limits are <=7151 MiB per run and nonincreasing reported evaluation loss.
The current whole-command target is <=27 seconds. Selected public v45 stays
selected until the measured candidate proves useful and satisfies those gates.

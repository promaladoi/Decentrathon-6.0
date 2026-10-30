# v44: lower-memory combined team candidate

Status: CPU validation passed; no v44 T4 performance, peak-memory or 10/10 claim.
The selected, measured public solution remains v34. This is an experiment.

## Implementation and attribution

Use Merei's corrected stage2 training path from exact public commit
`6a4f577e95605547b1b713013db1ffc44aa620bd`, directory `Мерей/solution`.
Its underlying Vladimir contributions retain their credit. Copy all 43 Python
files, including every existing test, then replace only package integration.
`04_submission/v44_provenance.json` records the 42 unchanged file hashes.
The old v43 source and tests remain in Git history and its public snapshot.

Nurzhigit contributes the existing exact frozen-storage implementation and a
new integration test. No multi-gigabyte decoded NF4 cache is used in this path.
The frozen tied embedding/LM-head Parameter is stored as FP16 only when every
finite FP32 value round-trips exactly. Unsupported or inexact weights retain
ordinary storage. Trainable LoRA parameters are never compressed. Embedding
outputs retain FP32; non-autocast public head calls retain FP32 arithmetic.
The Parameter identity and state-dict keys remain, but its internal storage
and corresponding state-dict tensor dtype become FP16; this is not a claim
of identical storage metadata. Values are exact.

The private LM-head consumes the same packed Parameter without another cached
FP16 cast. For the 151936 x 1536 tied matrix, removing the persistent FP32 copy
and duplicate half cast can reduce live storage; actual allocator peak, loading
peak and command time must be measured. No predicted MiB is a benchmark result.

Merei's private trainer context keeps ordinary public logits as a Tensor.
Failures after partial fast execution propagate instead of replaying dropout.
His actual-loss-scale head, tokenization map and metadata snapshot are retained.
The integration wrapper preserves the private-forward marker and original
exception identity. All package installation/import work remains timed.

## Validation and limits

Windows CPU: 50 tests passed, 9 skipped, 9 subtests passed. This includes the
opt-in real native package-metadata setup-prefix parity test. The raw log,
hashes and source digest are in `03_validation/reports/v44_cpu/`.
GPU-only tests were skipped locally; none is claimed as passed.

The Kaggle worker first runs strict dropout, storage integration and native
metadata parity tests, then the complete solution and Soup suites. Integration
uses the unchanged team test tolerances for real trainer metrics, rewind rows,
LoRA gradients and public Tensor outputs. Any failure stops before timing.

Next it uses the original harness for two paired base/solution runs and all ten
checks. The fixed model, NF4 configuration, data, steps, batch size, optimizer,
allocator, test thresholds and harness are unchanged. Additional acceptance
requires every solution peak <=7151 MiB and nonincreasing reported paired eval
loss. A 30-second target is checked separately and remains unachieved until a
complete-command T4 report establishes it. Do not compare only train-loop time.

Promote this snapshot only after verifying its exact source archive and actual
checks/result files. Never transfer Merei's or earlier Nurzhigit measurements to
this combined source. Smaller memory may trade off speed; judge both metrics.

# Follow-up optimization candidate

Based on `d6dca3ac371b6bf5ac15584cd9afbd027aab4a0b` in this repository.
The previous measured solution remains available at that commit. Its reported
56.09 seconds / 5275 MiB are not measurements of these changes.

## Changes

- Dropout saves eight real-token mask bits per byte instead of a boolean per
  element. Packing shares the existing gather/apply kernel. The native full
  padded random draw, dispatch threshold, multiplication order, FP32 forward
  and backward scales, exceptional-value behavior and fallback are preserved.
  Saved mask storage is `ceil(real_elements / 8)` bytes; this is not a claim
  about total GPU peak memory or training speed.
- Chunked LM-head statistics write entropy, argmax and labelled-token CE
  directly to their final row positions. This removes temporary metric arrays
  and their scatter launches. Prompt-only chunks do not gather targets or
  allocate CE/LSE. The online arithmetic, GEMM shapes, chunk size, loss
  denominator, actual upstream gradient scaling and public fallback stay the
  same. The old dense-output kernel API remains available for parity tests.

All modifications are under `solution/`. No organizer configuration, harness,
regression tests, data, batch order, optimizer or training switches are changed.
New kernels were written for this candidate; no third-party kernel library or
kernel source was introduced. Existing team attribution remains in the files.

## Validation status

- Python syntax compilation and whitespace checks pass locally.
- Added GPU checks for all 256 mask byte patterns, tail bytes, block boundaries,
  sentinel bounds, noncontiguous upstream gradients, scatter row/target strides,
  prompt outputs, vocabulary tails, exact dense/scatter parity and actual or
  mismatched AMP scales. Existing RNG, signed-zero and retained-graph checks
  remain in place.
- Full Kaggle T4 solution tests, Soup regressions, paired training/evaluation
  and the ten official checks are pending. No speed or total-memory gain is
  claimed before those measurements finish.

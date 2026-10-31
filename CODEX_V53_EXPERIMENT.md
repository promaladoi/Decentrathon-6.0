# v53: extend v45 lossless packed dropout masks to small batches

Exact starting source: selected v45, `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`.
v51 target-partition changes and v52 GEMM prototypes are not enabled.

One dispatch policy changes: all supported batch sizes use the already-qualified
native full-shape dropout RNG plus real-row gather and bit-packed saved masks.
Previously, padded tensors below 4*384*1536 elements retained the dense native path.
The value/gradient arithmetic, dtype, RNG draw shape/order and all module/layout
fallback guards stay the same. Compile-only warmup covers both full-row scalar
specialization classes inside the measured command.

Tests compare the actual small-batch dispatcher with dense native dropout,
including bitwise values/gradients, following RNG draws, saved-mask size and
hook observability. The full inherited numerical/Soup suites and original
two-base/two-solution harness remain required. All fifteen organizer files stay
unchanged. No measured speedup, peak-memory saving or 10/10 claim is made yet.

The target is <=27 seconds for the whole command, every solution run <=7151 MiB,
nonincreasing reported eval loss and original 10/10. A tradeoff or regression
will be recorded rather than promoted. Public selected v45 is preserved until
the new result establishes an advantage under the user constraints.

# v57 shared-memory Tensor Core research

Selected runtime remains exact measured v45. A profile of that implementation
places most GPU time in native dense GEMMs; the v52 global-load prototype was
slower. This next original prototype stages FP16 A/W tiles in shared memory,
uses an XOR word layout and explicit SM75 m16n8k8 FP32 accumulation. Matrix
dimensions are dynamic, avoiding per-batch shape compilation.

The first two compilation failures are retained with exact sources under
`03_validation/reports/v57_research` and `v57r1_research`. The next revision
uses explicit 64-bit addresses and packed b32 MMA fragments. A finite GPU
screen is pending: partial-row/column numerical cases first, then alternating
native/prototype timings. This code is not installed in solution.apply.

No full-task speed, memory, original 10/10 or <=27-second result is claimed.
Any training integration still needs complete native-gradient tests, all
original checks, <=7151 MiB and nonincreasing reported eval loss.

## Measured progress, still inactive

v57r2: eight numerical and eight timing cases passed numerical gates, including exact sampled native FP16 outputs. The 128x64 tile measured 1.682 vs native 0.748 ms for 512 rows; it was not selected.

v58: register prefetch and warp matrix loads also passed all eight numerical cases, but remained slower (1.469 vs 0.751 ms for 512 rows). No training integration.

v59: shared-address arithmetic is hoisted; 128x128/eight-warp and other geometries are being screened. This is not a complete-task score or a change to selected v45.

# v21: fewer startup compilations and shape-selected NF4 (T4 pending)

The explicit user target is **20–30 seconds for the same full command**, from
the observed 85.07-second base: 64.7–76.5% less total time, 2.84–4.25x speedup.
This target has not been achieved. The verified v16 remains 60.41 seconds.

The v16 median training loop was 36.30 seconds, leaving 24.11 seconds outside
the loop. The first two steps took about 5.70 seconds; the remaining 62 took
30.60 seconds. Therefore the target needs both startup and steady-step changes.
The 24.11-second remainder includes imports/load/setup/save/exit; it is not a
measured model-loading duration and must not be described as an immutable floor.

This revision combines v20 storage/mask work with Vladimir's latest warmup
revision `987da36` (fully credited in TEAM_INTEGRATION.md). It shares the head
gather/scatter binary between directions, reuses the add-norm binary at layer 0,
compiles in first-use order and waits for the in-process compiler before step 1.
No prebuilt compiler cache, timing boundary or task setting is changed.

The actual v17r1 T4 decoder microbenchmark showed paired-byte reconstruction
slightly faster on attention projections (~1.024x and ~1.027x) but slower on
the large MLP shapes (~0.939x and ~0.912x). The weighted decoder ratio was
0.9564x. Hence v21 uses the paired path only up to 4194304 weight elements and
retains the original shared decoder on larger shapes. Both are warmed inside
the same measured process. These decoder figures are not training speedups.

The fresh Kaggle session initially lacked bitsandbytes/Soup dependencies. The
failed v17/v18/v20 preflight archives are preserved; r1 jobs reuse the exact
source commits after restoring the same pinned packages. The dropout/storage
CUDA tests ran successfully in the first attempt, but no initial attempt
completed training. Full r1 measurements are still running while v21 is prepared.

This revision requires the full GPU suite, unchanged Soup regressions, paired
training/eval and all ten rules before admission. No v21 speed/quality result
is claimed by this document.

# v45: reduce transient head overlap and prepare the storage kernel earlier

Candidate only. v44 is pinned to its own source/queue and is unaffected by this
change. No v45 full-task time, peak GPU memory, quality or 10/10 is claimed.
The selected public production remains the measured v34 source.

## Memory lifetime

After a head chunk's statistics and optional gradient are consumed, release
its local logits reference before the next GEMM. Python normally retains the
old value while evaluating the right-hand side of the next assignment, which
temporarily overlaps two large allocations. The private precomputed path and
prompt rows need only the current chunk. The general autograd path deliberately
keeps the original supervised tensors in saved_chunks; retained graphs still
own them. Apply the same release to the scale-mismatch recomputation loop.

No GEMM size, order, data type, random draw, reduction, loss scaling, rounding or
kernel arithmetic changes. CHUNK remains 512. One full FP16 block of the fixed
151936-token vocabulary is 148.375 MiB. That is an allocation-size calculation,
not a measured whole-task saving: the overall peak may occur elsewhere.

An independent CPU ownership probe executes the actual _stats function from
both v44 commit 2d171ddc1f6d43e593f5dd80e1a86731f27ad6f7 and this source, with
only GPU arithmetic replaced by no-retention stubs. Before each successive
GEMM, v44 retains [0,1,1,1] previous temporary chunks; v45 retains [0,0,0,0].
The saved-backward path retains [0,1,2,3] in both versions and the GEMM row
schedule remains [3,3,3,2]. This proves Python ownership, not CUDA numerics.

## Startup

The own exact frozen-storage proof kernel was missing from v44's compilation
list. Compile and load its handles first in the existing in-process warm-up
thread while native setup loads the model. Nothing is launched by warm-up,
no random numbers are drawn, and no compiled cache is imported from another
run. All work stays inside the timed command. The first real launch still
compiles normally if the ordinary warm-up attempt failed. Actual startup gains
require measurement; no seconds are predicted from the source change.

## Validation

Targeted Windows CPU suite: 8 passed, 5 CUDA tests skipped, 2 subtests passed.
All 15 immutable organizer source files match their original SHA256 hashes.
Raw evidence is in 03_validation/reports/v45_cpu. The unchanged v44 tests are
retained. CUDA lifetime tests additionally exercise true kernels, native loss
and gradient parity, scale mismatch, and a second retained-graph backward.

The next Kaggle worker first runs head lifetime and existing strict scaled-head
tests, then all solution and Soup tests. It must complete two paired full-task
measurements, all ten original checks, each solution peak <=7151 MiB and no
increase in reported paired eval loss. The <=30-second total-command target is
tracked separately. Do not replace the selected production with this candidate
before original result/source verification. Team Merei/Vladimir authorship is
preserved; v45's small ownership/startup adaptations are by Nurzhigit/Codex.

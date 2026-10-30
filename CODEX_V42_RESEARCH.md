# v42 preparation: inactive gate/up matrix-forward fusion

**No T4 result is claimed. No change to the active solution.**
The v41 raw archive remains unavailable locally while the browser connection is
closed. Its last observed partial run was 44.31 seconds / 7011 MiB, with 330
solution tests and 79 Soup tests passed; the final paired result and ten original
checks have not been collected or verified. This is not an accepted v41 result.

The next experiment shares an input tile across two independent frozen FP16
projections, rounds both matrix results to FP16, and computes the existing
organizer SwiGLU forward formula in that same launch. It writes the same three
G/U/H buffers and adds no persistent decoded-weight cache. The original
activation formula is credited to the organizer; it is not presented as our new
SwiGLU algorithm. The matrix fusion implementation is new, with grouped tile
ordering informed by the [official Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

All code is isolated under `02_kernel_research/`. It is neither imported nor
enabled by `solution.apply()`. The older failed backward-fusion research and its
strict tests are retained unchanged.

The bounded GPU screen first runs the new numerical tests, then measures four
configurations on three task-representative row counts. A correctness failure
stops before timing. There is no production autotuning or cross-process cache.
The screen baseline uses the original organizer Triton activation with native
GEMMs, and must be measured separately from the full competition task.
Local validation was syntax compilation only; all 15 GPU tests were skipped on
the Windows CPU environment. This is explicitly not a GPU-correctness pass.

Before activation: successful T4 value/gradient checks, a real microbenchmark
gain, then two paired full runs with the original harness, all ten checks, every
solution peak <=7151 MiB and non-increasing reported paired eval loss. The
30-second complete-command target remains unmet and the selected production
snapshot remains v34.

If this research later qualifies, activation requires a separate measured
revision with the new implementation under `solution/`. Research-directory
launches are never counted as production own-kernel time.

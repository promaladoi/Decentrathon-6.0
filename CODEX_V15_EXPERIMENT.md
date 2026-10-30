# v15: repair dropout equivalence and compile active kernels at model startup

The participant's T4 execution of v14 `fd5e482` passed 106 tests but failed all
four exact dropout equivalence tests. The two implementations consumed the same
RNG state but assigned different masks to elements. No benchmark or adapter was
produced. The complete supplied console log is preserved in
`03_validation/reports/codex_v14_failed/participant_console.txt`.

Repair code `3016d55` obtains the mask from the same public
`torch.nn.functional.dropout` operation, full shape and dtype as the original.
A tensor of ones makes its nonzero outputs an exact boolean mask. Its rounded
fp16 scale is discarded; the compact kernel applies the original fp32 scale.
The exact-value, exact-gradient and exact-RNG tests keep zero tolerances, and two
successive-call tests check that the later random sequence stays unchanged.

Warmup code `ef10dd1` registers the active parallel-head implementation before
Vladimir's existing startup compilation thread begins. That thread now compiles
the actual head and compact-dropout signatures instead of the superseded serial
head kernel. It runs inside each measured training process while Soup loads the
model. Normal kernel launches, arithmetic, timing, allocator, task and harness
are unchanged; no prebuilt cache is loaded. Failed warmup safely falls back to
normal first-launch compilation, as in the existing shared implementation.

**v15 is not GPU-validated and has no claimed speedup, memory ratio or 10/10 pass.**
Local Python syntax and whitespace checks pass. GPU compilation and all numerical
tests must pass before two paired training runs and the ten rule checks. The
latest accepted console result remains v13, 1.398x / 5761 MiB / 10 checks.

`04_submission/run_candidate.py` automates that sequence and stops before training
on a failed correctness test. It first checks the untouched organizer file hashes
and idle T4 state. It saves exact source files and logs even on failure. The runner
does not edit the benchmark or manufacture result/check JSON from console output.

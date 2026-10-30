# v16: correct index storage layout before compact dropout

v15 `389f037` failed all six dropout tests and stopped before training.
Its failure archive was downloaded from the participant-provided link: all 31
Python source files match that Git commit byte-for-byte. The original test log,
outcome and archive hash are under `03_validation/reports/codex_v15_failed/`.
The signed archive URL was redacted from the participant console before saving.

The kernel reads row indices as `ROWS + row`, which assumes stride 1.
The failing tests pass `torch.arange(...)[::3]` and `[::2]`, whose storage stride
is 3 or 2. The kernel therefore read other row values and applied the wrong
tokens' masks. This is an indexing error. The earlier attribution to different
native/public dropout RNG mappings was not established and is withdrawn.

v16 makes `rows` contiguous once in forward, before the kernel call, and retains
that same tensor for backward. The production rows from nonzero already have
unit stride, so this normally requires no copy. Public dropout generation,
probability, tensor shape, arithmetic and RNG state are unchanged from v15.
Two new regression cases compare strided and contiguous storage of identical
indices for both dtypes, checking exact output, gradient and RNG-state equality.
All earlier zero-tolerance comparisons against original padded dropout remain.

v16 retains v15 startup compilation. The runner first executes 8 dropout tests,
then all 115 kernel tests, Soup regression tests, two paired training runs and
the ten organizer rule checks. Any test failure stops before training.

Local syntax/whitespace checks pass. **GPU validation and speed/quality results
for v16 are pending.** The accepted candidate remains v13 (1.398x, 5761 MiB,
10/10 console report). No 70–90% gain is claimed.

# v43: native package metadata reuse, with original import order

Candidate only; no T4 timing, memory, loss or 10/10 result is claimed.
The selected production remains v34. v43 builds on the v41 candidate whose
participant-reported paired measurement was 44.61 s / 7011 MiB / 10/10.

The v34 diagnostic identified two native packages_distributions() scans in one
process. Transformers already owns the original mapping when TRL repeats that
scan. Integrate teammate Merei's guarded snapshot helper and its existing tests,
unchanged, from collaboration revision
`6a4f577e95605547b1b713013db1ffc44aa620bd` under `Мерей/solution/`.
This is team reuse, credited to Merei, not independent kernel authorship.

Nurzhigit's integration defers installation until the native Soup SFT module
finishes its original import. A single-use finder recognizes the exact original
Soup 0.77 source; its native SourceFileLoader executes the unchanged body.
The native loader identity is restored before execution. No eager Soup/TRL import
is introduced. A version/source/provider mismatch retains ordinary behavior.
The teammate helper temporarily routes only TRL's native module-level metadata
call to a deep copy of the already-computed native mapping, then restores the
original stdlib function. It changes no availability/version result.

No GPU kernel, tensor arithmetic, training seed, model, optimizer, batch,
compiled cache or GPU memory cache is changed. All work remains inside the
timed command. The additional saving and 30-second target require measurement;
Merei's earlier timings are not presented as this combined candidate's result.

Before GPU tests/training, the worker runs a separate CPU-only native import
proof. It must show that the real deferred hook was reached, native setup ran
once, the actual TRL mapping equals a fresh native scan, the dict/lists are
independent, the stdlib function was restored and execution stopped before
training seed/model construction with CUDA uninitialized. A failed proof stops
the experiment and is archived, not bypassed.

Then the unchanged organizer harness runs two paired full launches per side,
all ten original checks, and the additional user limits. Every solution peak
must fit 7151 MiB and reported paired evaluation loss must not increase. A
sub-30-second result is not claimed until the complete-command measurement
shows it. Existing GPU tests and the older rejected research remain intact.

Local validation on Windows / Python 3.12 / Torch 2.11 CPU: 40 targeted tests
passed, including the teammate's opt-in native installed-package parity test.
The separate combined CLI/deferred-hook native prefix proof also passed with
zero repeated scans during setup and CUDA uninitialized. This verifies CPU
semantics in that environment, not T4 performance, GPU peak or hidden data.

# v71: qualify the RNG consumer independently of the AMP scale

v70's live original metrics showed the new Philox kernel only in compile
warmup, not timed training. A gradient-scale snapshot was an unnecessary
condition for dropout: it is needed for scaled head-gradient math, not RNG.
v71 checks the existing exact private-consumer scope, preserving all thread,
capture, version, T4, layout and large-shape guards. Added qualification covers
both absent and present loss scales with exact outputs, residual/parameter
gradients, RNG progression and retained graphs.

All source math, task, optimizer, data, steps, CLI and original 15 protected
files stay unchanged. Every solution test and original Soup test still runs.
The optional validation flag avoids rerunning individual test subsets before
that complete solution suite; it never skips the complete suite, original
paired harness, original ten checks or source/hash preflight.

Latest requested target: whole command 20 seconds and 500 MiB. Neither is
claimed before measurement. The original 10 checks and reported-quality rule
must pass; prior absolute 7151 MiB safety cap remains recorded separately.
This is the final candidate in the 40-minute optimization window.

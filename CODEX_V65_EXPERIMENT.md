# v65: prioritize whole-command time with bounded resident frozen weights

Starting implementation: exact selected v45, commit `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`.
No v64 rematerialization is included. The user prioritizes seconds; the earlier
hard peak limit of 7151 MiB and all original quality/task checks remain.

The admitted v45 GPU profile attributes about 10.94% of active GPU time to NF4
decoding. This candidate reuses exact decoded FP16 matrices in internal
read-only forward/backward consumers. Native small-M GEMM dispatch, native
GEMMs, the state-dtype rounding boundary and the fresh public decoder contract
are preserved. This is a deliberate time/memory tradeoff, capped at 2.4 GB of
resident decoded matrices; the original whole-task measurement must still
confirm every run stays <=7151 MiB.

The cache adapts the team's v39 implementation (`0a05a86`), adding cached-output
mutation checks and CUDA stream isolation. Normal source/state mutation,
replacement, metadata changes and unversioned inference tensors invalidate or
decline reuse. Entries use weak ownership and never retain packed aliases.
It is empty in every measured subprocess; initial decoding/allocation remain
inside the timer. No compiler cache, model or adapter crosses runs.

CHUNK stays 512. All task settings, 64 steps, optimizer, seeded dropout draws,
data order and Soup metrics remain unchanged. No new runtime, memory or
quality result is claimed until the original full paired harness and 10/10
checks finish. Current whole-command target: <=25 seconds.

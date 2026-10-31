# v64r2: preserve legacy LoRA helper calls without RMS metadata

v64r1 passed all eleven rematerialization qualification tests. The full team
suite then passed 392 tests and failed one existing LoRA-scaling compatibility
fixture: the attention helper supplied the new `norm_descriptor=None`
keyword to an existing replacement helper accepting the original signature.
No model training or original ten-check result occurred.

This revision keeps the old helper calls exactly when the RMS descriptor is
absent. The private rematerialization branch still receives the same metadata
and performs the same arithmetic as v64/v64r1. The old scaling test is kept
unchanged and now runs together with the eleven new tests before the full
suite. No RNG, training setting, original organizer file or original check
is changed.

The candidate is not selected. New archives use `codex_v64r2.zip`, preserving
both earlier failures. The additional user target is <=25 seconds for the
whole command, memory <=7151 MiB, nonincreasing reported evaluation loss and
all ten original checks. No new runtime, memory or quality claim is made
before the full original paired benchmark completes.

# Current measured status

Status: `not_selected`. Selected production remains v45.
[Verified report](../../CODEX_V46_RESULT.md).

The design note below is historical.

---
# v46: fold the LoRA input cast into the dropout producer

Candidate based on verified v45 (7a3db8fd5e24bbcb58affb475007a5cf93c56f07).
No new full-task performance or admission result is claimed before measurement.

In the qualified private packed path, native Linear under FP16 autocast casts
the FP32 dropout result immediately. The existing team gather/apply kernel now
can store that final FP16 value directly. Its multiplication order stays FP32;
the final storage cast is the same cast native Linear would receive. Backward
reads the half upstream as FP32 and writes the original FP32 input gradient,
without an intermediate half rounding. Native full-shape dropout still owns
every RNG draw; no state is manually advanced in production.

Only unobserved native Linear under CUDA FP16 autocast requests the half output.
Default/public dropout still returns FP32. Hooks and unsupported calls retain
the ordinary input dtype. Kernels and native arithmetic credit Merey's stage2
and the underlying Vladimir integration; this change removes conversion buffers.

Tests compare native RNG advancement, bitwise outputs and FP32 gradients,
noncontiguous upstreams, retained graphs, two shared-input LoRA branches and
trainable A/B gradients, plus hook-visible dtype. The untouched original task,
full solution suite and ten admission checks remain required.

Rejected alternatives: smaller head chunks failed the strict synthetic gradient
screen on some shapes and were slower on qualifying shapes. NF4 launch tuning
did not show a convincing architecture-wide gain. Neither alters production.

The target remains <=30 s whole command, <=7151 MiB, nonincreasing reported eval
loss and 10/10 original checks. v45 stays selected until a better qualified result.

# Current measured status

Status: `superseded_before_measurement`. Selected production remains v45.

The design note below is historical.

---
# v47: compile the used dropout signature and shorten temporary lifetimes

Candidate following v46. No full-task performance claim until measured.

The private FP16-autocast LoRA path uses the new half dropout producer.
Precompile that pair only; the FP32 public variant compiles on demand, as
ordinary Triton does. Compilation stays within the same timed process and
never imports another run's cache.

Release Python references to projection outputs after RoPE, attention output
after residual normalization, old MLP output after its residual addition, and
normalization output after MLP. Saved autograd tensors retain their normal
ownership; no saved tensor is modified. Operations and rounding are unchanged.

An instrumented/replaced Linear.forward keeps the original FP32 input dtype,
as do forward hooks. An extra CUDA test checks that observable behavior.

Verified v45 stays selected pending the original ten checks, the <=7151 MiB
and nonincreasing eval loss constraints, and an actual measured improvement.

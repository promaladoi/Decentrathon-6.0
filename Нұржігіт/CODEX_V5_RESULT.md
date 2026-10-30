# v5: first validated result above the 5% speed threshold

Code commit: `58c73787a7b01186a437a82854f91db9b7db7646`.
The frozen Qwen MLP retains its intermediate activations in the active fp16
autocast compute dtype. Separate input casts preserve accumulation into the
original float32 input. The organizer's SwiGLU remains attributed and excluded
from our own-kernel count. Internal gradient rounding can differ; task settings
and model/data/optimizer are unchanged and quality was checked end to end.

Kaggle T4, two paired runs: base times **79.46 / 86.77 s**, solution times
**77.34 / 78.94 s**. Harness summary: **83.11 / 78.14 s**, speedup **1.0636x**,
approximately **6.0% less time**. Baseline timing varied, so this is the measured
two-run median result, not a guarantee for every run or GPU session.

Peak memory: **14345 / 10345 MiB**, about **27.9% less**. Public eval loss:
**2.04964 / 2.04963**. Own kernels: **7.9%** minimum across the two runs.
All **10 checks**, **37 existing kernel tests**, and **79 Soup tests** passed.
The existing unit tests cover the individual kernels; the new MLP integration's
numerical validation is the full loss/quality/Soup-output check in the harness.

Raw reports under `03_validation/reports/codex_v5/` preserve their original bytes.
Report hashes and exported solution files were verified against the measured
commit. Reproduce with `templates/codex_v5.ipynb` on T4 with Internet On.
The active Kaggle draft still has save conflicts; downloaded artifacts and Git
history preserve the experiment independently. Final public submission is pending.

The subsequent v6 experiment `f92ebf2` fuses NF4 reconstruction with widening
for backward, preserving the intermediate fp16 rounding. Its measurement is
in progress; this document makes no performance claim about v6.

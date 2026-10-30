# v6: fused NF4 backward reconstruction and widening

Measured code: `f92ebf2ecb1f3f974d6e42115400b20d741dd124`.
When backward needs float32 reconstructed weights from an fp16 QuantState,
the decoder preserves fp16 rounding and writes the widened value directly.
This removes the intermediate allocation and copy. Other cases retain the
original backward. This includes the v5 MLP optimization.

Kaggle T4, two paired runs: base **86.21 / 87.11 s**, solution
**78.29 / 78.80 s**. Summary **86.66 / 78.55 s**, speedup **1.103x**,
about **9.4% lower runtime**. Memory **14345 / 10345 MiB** (27.9% lower).
Public eval loss **2.04967 / 2.04967**. Own-kernel minimum share **7.9%**.
All **10 checks**, **37 kernel tests** and **79 Soup tests** passed.

Raw reports and solution source hashes were verified after downloading.
Reports: `03_validation/reports/codex_v6/`; notebook: `templates/codex_v6.ipynb`.
Baseline times vary between experiments. v5 and v6 solution times are similar,
so these paired ratios alone do not isolate the benefit of the new widening path.
No claim is made of maximum performance or hidden-data quality.

The next v7 experiment adds original LoRA scale/add forward and backward
kernels, preserving the operation's intermediate rounding, dropout and linear
modules. v7 is still being measured.

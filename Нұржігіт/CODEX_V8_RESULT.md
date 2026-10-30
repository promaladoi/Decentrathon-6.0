# v8: direct half-logit loss reduces memory

Code: `7435a924e0711919b46487084967274b2c495399`.
The team vocabulary reduction reads fp16 logits and performs arithmetic in
float32, preserving causal label shifting, masking, normalization and fp16
gradient output. This removes the full float32 logits copy inside causal loss.
It is based on the semantics of Transformers 5.16.1 ForCausalLMLoss (Apache-2.0).

All **47 kernel tests**, **79 Soup tests**, and **10 harness checks** passed.
Kaggle T4, two paired runs: base 85.91 / 86.36 s; solution 84.86 / 84.96 s.
Summary base **86.13 s**, solution **84.91 s**, speedup **1.014x** (speed tie).
Peak memory **14345 / 9143 MiB**, about **36.3% lower**.
Public eval loss **2.04963 / 2.04967**, delta **+0.002%**.
Minimum own-kernel share **21.4%**.

This experiment still includes the slower v7 LoRA fusion. It demonstrates the
loss change's memory benefit but is not the preferred speed candidate. v9 removes
that LoRA patch and combines the loss change with the faster v6 implementation.
v9 remains under measurement.

Original reports: `03_validation/reports/codex_v8/`. Report bytes and exact
solution source were verified against their hashes and measured code commit.
Reproduction: `templates/codex_v8.ipynb`. No hidden-row performance claim is made.

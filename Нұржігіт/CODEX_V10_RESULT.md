# v10: preserve half logits, retain float32 reductions

Code: `e19b464b824bb1eb910217cac22b830a25207e39`.
The Accelerate output conversion retains the original fp16 vocabulary logits.
Other output fields follow the original conversion. Loss, entropy and rewind
cross-entropy reductions load into fp32; accuracy uses the same logit ordering.
The model, data, training settings, optimizer and harness are unchanged.

Two paired T4 runs: base **86.27 / 86.92 s**, solution **77.25 / 78.00 s**.
Median **86.59 / 77.62 s**; speedup **1.1156x**, about **10.4% less runtime**.
Peak memory **14345 / 7385 MiB**, **48.5% less**. Eval loss **2.04968 / 2.04966**.
All **10 checks**, **54 kernel tests** and **79 Soup tests** passed.
Minimum original-kernel share: **7.0%**. Exact source and raw-report hashes verified.

Reports: `03_validation/reports/codex_v10/`. Reproduction: `templates/codex_v10.ipynb`.
The speed difference from v9 is small and may include measurement variation;
the memory improvement is substantial. This is not a hidden-data guarantee.
v11 and v12 are subsequent experiments, not yet accepted candidates.

Source semantics inspected: Accelerate 1.14.0 `utils/operations.py`,
TRL 0.29.1 `trainer/utils.py`, Transformers 5.16.1 causal output/loss.
All reduction kernels are team implementations; the organizer SwiGLU stays attributed.

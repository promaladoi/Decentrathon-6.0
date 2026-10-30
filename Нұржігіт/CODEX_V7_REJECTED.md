# v7: valid but slower; not selected

Code `a044007717f44005fd3d85dfc571a940e1051c91` added fused LoRA scale/add
and its backward. All 43 kernel tests, 79 Soup tests and 10 harness checks
passed. However, the complete T4 benchmark showed no useful speedup:
base runs 86.22 / 87.16 s; solution runs 86.66 / 86.81 s.
Peak memory was 10343 MiB versus base 14345 MiB. Public eval loss was
2.04962 versus base 2.04966. Own-kernel share rose to 22.0%; that is not
itself an optimization objective, and the end-to-end runtime regressed
relative to the earlier v6 solution times around 78.5 s.

The LoRA patch was removed in `57182a2`. Its implementation and tests remain
in Git history. Unmodified reports are under `03_validation/reports/codex_v7/`;
their result hash and exact code snapshot were verified after download.

v8 evaluates direct half-logit causal loss on top of this version; v9 isolates
that loss change on the faster v6 implementation. Both are still in progress.

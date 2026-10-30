# Codex v4: fp16 autocast weight-buffer experiment

Code commit: `12b9f58f32e70fc1db6ca47b7860eb9870a9ffd8`.

The existing NF4 forward fallback now decodes directly to fp16 when its
`F.linear` call already runs under CUDA fp16 autocast with float32 inputs.
This avoids creating a float32 decoded-weight buffer that autocast immediately
converts to fp16. Input and bias casting remain in the same `F.linear` path.
Other cases retain v3 behavior. No task or measurement settings changed.

Tests compare forward results and input gradients with the original bitsandbytes
backend, with/without autocast, fp16/fp32 inputs and nested/plain NF4 scales.
All comparisons use zero tolerance. On Kaggle T4, **37 kernel tests passed in
6.06 s**, and **79 Soup tests passed**. There were 14 upstream deprecation warnings.

## Actual paired measurement

Unchanged harness, two repeats in base/solution/solution/base order:

| Run | Whole training time, s | Peak GPU memory, MiB |
|---|---:|---:|
| base_1 | 85.77 | 14345 |
| solution_1 | 83.23 | 12385 |
| solution_2 | 83.13 | 12385 |
| base_2 | 86.57 | 14345 |

| Summary | Base | v4 |
|---|---:|---:|
| Median whole training time, s | 86.17 | 83.18 |
| Peak GPU memory, MiB | 14345 | 12385 |
| Public eval loss | 2.04967 | 2.04968 |

**All 10 checks passed.** Own-kernel share: **7.1%**, lowest of two solution
runs. Speedup: **1.0359x**; memory ratio: **1.1583x**; reported eval loss delta:
**+0.000%**. Runtime is about **3.5% lower**, below the competition's 5% speed
threshold, so speed still counts as 1.000x. Much of the memory saving comes
from the retained organizer SwiGLU example.

This does not demonstrate a meaningful improvement over v3's 1.0326x ratio.
Absolute seconds from separate experiments must not be used to claim that
v4 is faster than v3. Both candidates pass the public checks; neither has been
compared with teammates' submissions. The v3 reproduction pin is preserved,
and v4 has its own notebook: `templates/codex_experiment_v4.ipynb`.

Raw reports: `03_validation/reports/codex_v4/result.json` and `checks.json`.
Their byte hash and the exact exported solution files were checked against
the code commit and checks report after downloading. The v4 PEFT adapter has
an 8,731,128-byte `adapter_model.safetensors`, SHA256
`b5a223f37693ce365adb5cca2a255449d2d02f9a8e51cc8db603f17c4e68c8ac`.

The unchanged public organizer starter was also run directly at commit
`c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`: four example tests and 79 Soup
tests passed; checks 3/4 failed as expected and the other eight passed.
Its raw reports are under `03_validation/reports/original_verify/`.

The active Kaggle draft has had save conflicts. The downloaded reports and
adapter are preserved independently; do not treat the live draft as a saved
public submission version. Final selection and a saved public Kaggle version
are still required before submission.

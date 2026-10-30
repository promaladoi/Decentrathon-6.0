# v9: validated Codex candidate

Measured code: `57182a2312784854529f08693a31a428ed8081c3`.
Kaggle T4, one GPU, two paired runs in base/solution/solution/base order.

| Metric | Base | v9 |
|---|---:|---:|
| Run 1 wall time, s | 86.94 | 79.12 |
| Run 2 wall time, s | 87.84 | 78.70 |
| Median full training time, s | 87.39 | 78.91 |
| Peak GPU memory, MiB | 14345 | 9165 |
| Public eval loss | 2.04966 | 2.04964 |

**1.1075x speedup** means approximately **10.75% higher throughput** or
**9.7% lower runtime**. Memory usage is **36.1% lower** (base/solution 1.5652x).
All **10 rule checks**, **41 kernel tests** and **79 Soup tests** passed.
Six original kernels account for at least **7.3%** of measured training time.
The unchanged organizer SwiGLU example is excluded from this share.

## Changes

v9 combines v6's fused NF4 reconstruction/widening and v5's guarded fp16 MLP
intermediates with v8's direct causal loss. Causal loss reads fp16 logits and
performs the vocabulary reductions in fp32, avoiding a full fp32 logits copy.
Causal shifting, ignore labels and normalization are preserved; unsupported
cases fall back to Transformers. The slower v7 LoRA fusion was removed.

NF4 widening explicitly preserves fp16 rounding before writing fp32 output.
The MLP change can alter intermediate gradient rounding; it is not claimed
bitwise identical. End-to-end loss curves, task outputs and public evaluation
passed the organizer tolerances. The fixed model, data, optimizer, LoRA settings,
steps, harness, task and organizer tests were not changed.

## Reproduce and inspect

- Import [the pinned notebook](templates/codex_v9.ipynb) into Kaggle.
- Use T4 x2 with Internet On; the harness uses one GPU. Authorize the existing
  GitHub secret for the private platform repository, or use the public mirror URL.
- Keep COMMIT fixed to the SHA above and Run All. The benchmark uses two repeats.
- Original reports: [result.json](03_validation/reports/codex_v9/result.json)
  and [checks.json](03_validation/reports/codex_v9/checks.json).
- After download, report SHA256 and exact solution source bytes were verified
  against checks.json and the measured Git commit.

The current interactive Kaggle draft reports a save conflict. Raw reports,
the trained safetensors adapter and exact source were downloaded independently;
the supplied notebook is a reproducible input, not a published executed version.
Create a successfully saved, pinned execution before final submission.

## Interpretation

This is the best current Codex candidate considering speed and memory together.
v6's 1.1032x and v9's 1.1075x ratios are close, so the small speed difference
between those variants is not established beyond measurement variation. v9's
memory reduction is larger. Earlier v7 and v8 results remain in history even
though their speed was worse. Two paired runs do not prove maximum attainable
performance, hidden-data quality or a winning score; other teams' admitted
results also determine points. Teammates' variants have not yet been compared.

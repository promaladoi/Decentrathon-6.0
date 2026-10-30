# Measured Merei optimization, 2026-10-30

Runtime source: `29419f43d9b9173d1ba5d6824a31589ad3b84df7`.
Final solution documentation: `8d0c83fa6a9e8cd31e344efc61f6410690fd87fe`.
All solution Python Git blobs are identical between those commits. The final
check_rules rerun binds the final documentation bytes and unchanged runtime.
Protected harness/task/tests are identical to current Merei d6dca3ac.

Current-Merei A/B (old,new,new,old): 52.30/54.46 -> 48.50/49.84 seconds;
medians 53.38 -> 49.17 seconds (-7.9%), max GPU peak 5275 -> 5175 MiB.
Timed steps 3-64 were 25.707 -> 25.737 seconds: no steady-state speedup claim.

The previous candidate 7e7330b produced 48.615 seconds in a separate series.
Direct 7e-vs-294 series: medians 49.645 -> 47.11 seconds, both 5175 MiB.
Do not combine medians from different series.

Required official admission on the selected runtime: all 10 checks passed,
339 solution tests passed before A/B, 79 Soup regression tests passed under
check_rules. Quality loss 2.04966 (delta -0.003%); own kernels 14.5% on steps
3+, excluding starter SwiGLU. Official solution repeats: 48.14/49.89 seconds,
peak 5175 MiB in both. Native runs are retained only as required admission proof.

`final-result.json` is the original, unmodified harness result. `final-checks.json`
binds the final solution bytes; `measured-294-checks.json` binds the measured
snapshot before the documentation update. `source-proof.json` records source
identity. The report files and this README are outside solution and do not
change its SHA256. Full executed notebook outputs were saved in Kaggle.

`manifest.json` covers this report directory. `kaggle-bundle-manifest.json`
records the original downloaded bundle, including eight identical downloaded
model tokenizer.json copies omitted from this report directory. Training logs,
metrics, configs, trainer state and original comparison JSON are included.

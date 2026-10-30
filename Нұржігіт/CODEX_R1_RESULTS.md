# Fresh-session r1 results: all exact sources and report hashes verified

The first v17/v18/v20 preflights stopped because the new Kaggle session lacked
bitsandbytes/Soup packages. Their original failed archives and logs are retained.
The r1 attempts use the **same exact code commits** after restoring the pinned
dependency versions. No failed outcome was overwritten or represented as training.

| Version | Source | Base / solution, s | Speedup | Time saved | Peak MiB | Memory saved | Eval delta | Checks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v17r1 | `7c5f6d8` | 84.28 / 54.19 | 1.5553x | 35.7% | 5601 | 61.0% | -0.002% | 10/10 |
| v18r1 | `d01b2d8` | 80.94 / 55.61 | 1.4555x | 31.3% | 4701 | 67.2% | +0.000% | 10/10 |
| v20r1 | `22ff279` | 81.23 / 55.81 | 1.4555x | 31.3% | 4467 | 68.9% | -0.000% | 10/10 |

All three use two base and two solution runs. 138 / 158 / 174 solution tests,
respectively, and 79 Soup tests passed. Every solution source file was compared
byte-for-byte with its exact Git revision; original result/source hashes and
archive CRCs were verified. Raw reports, test logs and collection proofs are in
`03_validation/reports/codex_v17r1`, `codex_v18r1` and `codex_v20r1`.

v17r1 currently has the highest collected paired speed ratio. v20r1 has the
lowest collected memory use. They are different time/memory tradeoffs; no final
team submission is selected. Separate-session/paired-run variation exists, so
do not attribute the full wall-time difference between v16 and v17 to the NF4
decoder alone. Its standalone large-shape microbenchmark was actually slower;
v21 therefore keeps the original decoder on those large shapes.

The current user target is **20–30 seconds for the full unchanged command**,
about 64.7–76.5% below the earlier 85.07-second base. It has not been achieved.
v21's console reports 10/10 at roughly 55.5 seconds; its archive collection is
pending. v22 startup imports and v23 bounded weight reuse are under validation.
Profiler runs are separate diagnostics and must never be reported as contest scores.

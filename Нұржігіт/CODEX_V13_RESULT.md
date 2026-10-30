# Combined v13: completed Kaggle console report

Code: `13eaa8a025752db94a020102fa3ecd90951a2fec`.
Source of this record: the participant copied the completed Kaggle console output
into this chat after the browser connection closed. This is a console report;
the raw `result.json` / `checks.json` archive has not yet been downloaded and
hash-verified locally. No replacement raw JSON has been fabricated from the text.

| Metric | Base | Combined v13 |
|---|---:|---:|
| Training run 1, seconds | 89.06 | 62.93 |
| Training run 2, seconds | 87.97 | 63.68 |
| Displayed median, seconds | 88.5 | 63.3 |
| Peak memory, MiB | 14345 | 5761 |
| Eval loss | 2.04968 | 2.04966 |

Reported speedup **1.398x** = **39.8% higher throughput**, about **28.5% less time**.
Memory ratio **2.490x** = **59.8% less peak memory**. Quality delta **-0.001%**.
**106 kernel tests**, **79 Soup tests**, and **10/10 rule checks passed**.
Own-kernel share: **11.3%** minimum of two solution runs, from step 3 onward.

The reported ten checks were:

1. PASS: all two base and two solution runs finished.
2. PASS: no banned kernel libraries; no packages changed during the run.
3. PASS: actual team Triton kernels ran in every solution run; organizer SwiGLU excluded.
4. PASS: own kernels took 11.3% of measured training time.
5. PASS: same task and Soup outputs within tolerance.
6. PASS: solution first-step loss difference 0.02%, mean 0.01%; base repeat 0.00% / 0.01%.
7. PASS: eval loss -0.001% relative to base.
8. PASS: time x0.715 and memory x0.402 of base.
9. PASS: clean T4 measurement, one GPU, required allocator, peaks agree.
10. PASS: 79 Soup regression tests, 14 warnings, 0.84 s.

This combines Vladimir's shared unpadded forward, fused normalization/rotary/NF4
operations and chunked head with Nurzhigit/Codex parallel tiled vocabulary
statistics. See TEAM_INTEGRATION.md for exact provenance. It is not an independent
reimplementation of Vladimir's contribution.

The participant's Vladimir result (1.468x, memory ratio 2.543x) remains better by
its reported ratios than this v13 result. Separate sessions and a single reported
peer pair do not establish which component is faster; a controlled comparison
is still needed. No 70–90% speed gain is claimed.

The initial v13 test run had two relocation import errors, then the corrected
code above passed all 106 tests. The original harness/task/tests remain unchanged.
v14 adds compact dropout and requires its own tests and full report before acceptance.

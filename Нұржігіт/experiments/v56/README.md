# v56: qualify a narrow NF4 dispatch opportunity from the v54 screen

Starting runtime is exact selected v45. v55's activation recomputation and
v53's dropout-policy changes are not enabled, so the speed change is isolated.

The pinned baseline bitsandbytes 0.50.2 supplies its own fused NF4 MMA kernel.
The team does not claim authorship of that kernel. The v54 inactive screen
showed it slower for large MLP projections, but faster for 256-output
projections. This candidate uses that existing backend only for contiguous
FP16 inputs, narrow/tall weights, M <= its native supported maximum, matching
FP16 bias and SM75. The remaining path and input-gradient arithmetic are v45.

This fused reduction need not be bitwise equal to unfused FP16 GEMM. New tests
use the existing FP16 2e-3 absolute/relative tolerance, plus bitwise native
input gradients. Existing strict tests are not weakened or removed. The
original loss-curve/evaluation checks and stricter nonincreasing user loss
still decide eligibility. Baseline library inventory is unchanged.

No complete-task speed or quality result is claimed before the paired
original harness. The <=27-second whole-command goal, <=7151 MiB memory
bound and original ten checks remain required. Selected public v45 stays.

[Vendor primary ABI](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py).

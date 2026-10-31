# v54 inactive native NF4 screen

Pinned bitsandbytes 0.50.2 already supplies this vendor fused kernel. This is
not team-authored kernel code. No new package was installed and production
was not changed. Twelve geometry cases were measured with alternating timing.

For the gate/up projection, 512 tokens measured 2.729 ms fused vs 0.915 ms
selected; 832 tokens measured 2.678 vs 1.001 ms. Down projections were also
slower and failed the screen's numerical tolerance. Do not force this path
for the expensive MLP operations.

The 256-output projection has a smaller opportunity: 832 tokens measured
0.089 vs 0.127 ms and passed this screen's tolerance, but was not bitwise
equal. 128/512 tokens were faster but did not pass the same tolerance. This
is not a whole-command speedup or a 10-check admission. A future change still
needs native forward/backward tests and the complete original evaluation.

Temporary allocation measurements are not the organizer's NVML peak.
27 seconds is not claimed. [Primary ABI](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py).

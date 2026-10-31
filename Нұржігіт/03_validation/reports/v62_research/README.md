# v62 exact NF4 geometry screen

36 shape/launch comparisons matched native bitsandbytes exactly. Same decoder,
same arithmetic, no production change or complete-task admission yet.

For 8960x1536: 1024 elements / 8 warps measured 0.151906 ms vs the v45
1024 / 4 baseline 0.160872 ms (~5.6% less kernel time). For 1536x1536:
2048 / 4 measured 0.029871 vs 0.031698 ms (~5.8% less). The smaller
256x1536 case had no material improvement. Native-exact output does not by
itself establish model quality or a whole-command speed benefit.

Graph replay amortized launch gaps for this isolated measurement only. No
graph or observer change is proposed for training. All alternating samples
are retained in screen.json. The 27-second goal is not claimed.

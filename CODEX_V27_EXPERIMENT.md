# v27: bounded reuse of all eligible frozen NF4 projections

Experimental on top of v26. v23's original admitted metrics show repeated small
NF4 decodes still launch 14,145 times over the run despite its MLP cache. v27
extends the same mutation-aware exact decoder cache to q/k/v/o matrices too:
minimum 262,144 elements, maximum 2.8 billion resident bytes. The first decode
and all cache construction remain inside each measured process. Model packed
parameters, NF4/double-quant state and decoded values are unchanged.

The projection cache trades additional VRAM for fewer decoding, allocation and
launch operations. It must still pass the original >=5% real own-kernel share,
idle/single-T4 rules, unchanged task, loss/quality and regression checks. No
extra GPU work is inserted to influence kernel share. A failed share or quality
check would reject this experiment in favor of an admitted earlier version.

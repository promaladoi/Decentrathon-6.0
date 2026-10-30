# v31: compare the small-projection cache on time and memory

Experimental balanced counterpart to v30's full 2.8 GB cache. The scoring also
awards memory efficiency, so shortest absolute time alone may not rank highest.
v31 keeps q/k/v/o decoded matrices (262,144 to under 4,194,304 elements) with a
400 MB budget, and decodes large MLP matrices on demand. It retains the same
exact-value mutation-aware process-local cache, not another model/quantization.

Both speed and memory will be compared in original paired runs; no claim that
this policy wins is made in advance. Small-cache routing has explicit CPU size
boundary and actual CUDA matrix reuse tests. v29 GEMM research remains inactive.

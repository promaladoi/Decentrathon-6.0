# v36: credited Merei helpers on the fastest native-GEMM path

Imports `backward_sums.py`, `batch_metadata.py` and their strict tests from
Merei's team source `29419f43d9b9173d1ba5d6824a31589ad3b84df7`, adapting only
warmup to our existing final launcher-preparation stage and package imports.
Authors are explicitly Merei and his development tools, not Nurzhigit alone.

All native GEMMs and their FP16 outputs remain unchanged. A team Triton kernel
casts those outputs to FP32 and sums in the original order in one pass. Another
validated kernel builds real rows and cumulative sequence lengths with one host
read; the native attention plan uses that metadata directly. Invalid metadata
falls back before dropout; unsupported sizes retain the ordinary row builder.
The full 2.8 GB frozen-weight cache and v28 startup improvements are retained.

Bitwise exceptional-value/ordered-sum tests and exact rows/boundaries/RNG tests
travel with these helpers. All active model tests, Soup regression tests and
ten unchanged official paired checks must pass before any speed claim. The
rejected independent GEMM research stays outside the production package.

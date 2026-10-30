# v34: production candidate using only established native GEMM

The inactive two-product research failed its revised per-entry numeric oracle
(8 failing cases); it was never used by any training path. Its implementation,
tests and benchmark are now together under `02_kernel_research/`, outside the
production solution package. The failed tests remain present and documented.
No active model/gradient test or organizer rule has been removed or relaxed.
Prepared v30/v31 were not run; v32/v33 stopped before training on research tests.

v34 retains native PyTorch matrix products and the admitted v28 path, adds the
strict unused-dense-mask guard, and retains the full 2.8 GB frozen-weight cache.
Its production suite and all ten unchanged paired checks must pass before use.

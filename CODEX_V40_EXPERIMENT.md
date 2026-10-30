# v40: restore the bounded head chunk after v38 exceeds the memory limit

Prepared for exact-source validation, not a speed claim.

The first complete v38 solution measured 44.37 seconds and **7171 MiB**. This is
20 MiB over the user's limit, so v38 must not replace production even if all ten
original checks pass. v39 inherited the same larger chunk and remains experimental.

v40 restores the proven 512-row head GEMMs while retaining prompt temporary
release, v37's byte-identical tokenizer canonicalization, and v39's cache-hit
metadata optimization. No decoded-weight budget is raised. Larger-boundary
regression tests remain and still cross the restored 512-row chunks.

Every solution peak must be <=7151 MiB, reported evaluation loss must not exceed
the paired base, and all ten original checks must pass. The 30-second full-command
target is still not achieved. v34 stays selected until a better qualified version
is actually measured.

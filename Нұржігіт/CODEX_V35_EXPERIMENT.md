# v35: production small-cache counterpart to v34

Uses the same admitted native PyTorch GEMM, v28 startup improvements and guarded
native variable-length attention. The cache retains only attention matrices
under 4,194,304 elements with a 400 MB cap; large MLP matrices decode on demand.
Unused rejected matrix-fusion research and its strict tests remain outside the
production package. All active production tests and ten original rules apply.

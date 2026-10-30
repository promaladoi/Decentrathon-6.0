# v33: lower-memory counterpart of v32

Prepared v31 policy, with the corrected inactive research tests, now as a new
exact code revision. Keeps only attention matrices below 4,194,304 elements in
the mutation-aware 400 MB cache; large MLP weights decode on demand. All v28
startup improvements and the native variable-length/dense-mask guard remain.

The candidate must pass the full solution suite, original Soup tests and all
ten original paired-run checks. It is not an admitted or measured result yet.

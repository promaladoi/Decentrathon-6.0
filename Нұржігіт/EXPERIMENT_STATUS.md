# Current experiment

v11 code 9e104d4 adds a frozen output-head cast cache and skips exactly zero
upstream-gradient rows in its backward GEMM. Tests and paired validation are pending.
v10 e19b464 passed 54 kernel tests and 79 Soup tests; paired measurement is running.
The pinned candidate notebook still runs validated v9, not these experiments.

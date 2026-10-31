# v53r1: qualify the intentional all-size dropout dispatch policy

Runtime code is identical to v53, built from exact selected v45. v53 stopped
before training: its small-batch numerical, gradient and RNG comparisons passed,
but an inherited team-owned routing assertion still expected the previous
384-token cutoff. No organizer check ran and no training result was produced.

v53r1 changes that routing expectation to two qualified Q/V calls at every
supported size, and expands the integration cases to 127, 128, 256, 383, 384
and 512 tokens. All inherited zero-tolerance value/gradient and exact RNG-state
assertions are retained. No organizer file, numeric tolerance or user limit is
changed. The full solution/native-parity/Soup suites must rerun before training.

No speed, whole-run memory improvement, 10/10 or <=27-second claim exists yet.
Selected public v45 remains unchanged during the experiment.

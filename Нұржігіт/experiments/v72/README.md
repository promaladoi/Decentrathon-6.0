# v72: remove the observed background renderer, retain strict RNG qualification

The original first-forward diagnostic on v71 identified `rich.live._RefreshThread`
alongside MainThread and native tqdm TMonitor. This was the rejection reason;
the private scope and AMP scale were already available. No thread is whitelisted
or RNG guard weakened. The exact native Soup dashboard start/update sources are
guarded by hashes. The same dashboard state and Panel are rendered synchronously
from the native update callback, rather than a background Rich refresh thread.
Metrics, model computations, dropout draws and original benchmark stay intact.
Unknown implementations and subclasses retain their original behavior.

Three CPU qualification tests verify all native state fields, sticky evaluation
values, rendering, lifecycle and unknown-source fallback. The complete solution
suite, original Soup suite and two-base/two-solution original admission remain
required, as do nonincreasing reported evaluation loss and the old 7151 MiB cap.
Latest requested objective is 20 seconds AND 500 MiB; no success is claimed yet.

Diagnostic memory evidence: the mandatory pre-solution CUDA context used 107 MiB.
Registered CUDA parameter storage alone used 1078.740234375 MiB. Model dimensions
give a 1069.875 MiB lower bound for NF4 matrices plus FP16 embedding, excluding
quantization metadata, gradients, activations, workspaces and CUDA overhead.
Even hypothetical four-bit storage for every matrix and embedding is 736.03125
MiB. These are payload bounds for resident weights, not proof against every
possible CPU streaming design; CPU offload would require a separate validated
implementation and transfer/compute measurements.

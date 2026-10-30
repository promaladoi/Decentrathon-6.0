# v38: one live vocabulary chunk, larger native head GEMMs

Status: unmeasured experiment; not the selected production version.

Extends v37. The parallel head explicitly frees each vocabulary-sized temporary
before the next chunk. It uses one 1024-row chunk instead of retaining consecutive
512-row buffers. Both native GEMMs and the full-vocabulary CE/entropy/argmax
computations remain. No rows, labels, optimizer work or Soup metrics are omitted.
The new regression crosses the larger chunk boundary and compares loss, complete
metrics and input gradients with the ordinary dense PyTorch head.

Temporary lifetime reasoning is not a claim that overall memory is unchanged.
The unchanged harness must measure **every solution run at <=7151 MiB**; otherwise
this candidate will not replace v34 even if all ten organizer checks pass.
The 30-second full-command target is not yet established.

# v39: validate frozen-cache hits without rebuilding metadata records

Experimental, not yet measured. Extends v38.

On a cache hit the same identities, weak references, mutation versions, storage
pointers, shapes, strides, dtypes, devices and scalar quantization settings are
checked directly against the saved signature. No decoded weights or GPU buffers
are added. On a miss the original snapshot/decoder path remains.

Existing invalidation/aliasing/budget tests are retained. Additional cases cover
`.data` replacement without a version increment and metadata changes after a
cached hit. The unchanged full-task harness and the extra 7151 MiB ceiling decide
whether this experiment is usable; no 30-second result is claimed.

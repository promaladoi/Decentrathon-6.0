# v32: full-cache native attention after research-test correction

Exact new snapshot of the fast v30 path, which was prepared but not measured
after the independent research cell aborted the notebook queue. Includes the
unused dense-mask guard and v28's faster complete tokenizer serialization.
All frozen NF4 matrices use the bounded 2.8 GB process-local cache.

The inactive GEMM research now has a partial-FP16-ulp error oracle plus an exact
cancellation case and a runtime token count. This does not enable that kernel
in training. Correctness tests still precede a completely unchanged paired
benchmark and all ten original admission rules. No v30/v31 result is claimed.

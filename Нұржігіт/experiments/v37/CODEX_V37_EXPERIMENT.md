# v37: bounded-memory continuation toward 30 seconds

Status: prepared for validation, not a measured speed claim.

Starting point: the previously admitted v36 team integration (v34 remains the
selected published production snapshot). Same fixed task and original checks.
The requested additional ceiling is **7151 MiB in every complete solution run**.
The requested 30-second full-command target has not been demonstrated.

The complete Qwen tokenizer fingerprint now canonicalizes large immutable leaf
maps with sorted item insertion rather than a recursive Python call per value.
The key ordering, object aliases, complete state and resulting pickle bytes are
unchanged. Container/custom-object maps retain the old recursive path. New tests
compare exact bytes with the prior implementation, including aliases/cycles.

Fresh v34 diagnostic (`codex_diagnostic_fe2272c.zip`) remains separate from contest
results. Its ordinary solution-only run was 42.02 seconds / 7151 MiB; it is not a
new paired admitted score. Background compilation already finishes before step
one, so extra compile threads are not introduced without a demonstrated need.

# v70: generate exact native dropout bits directly for real rows

Candidate awaiting full original admission; no elapsed-time improvement is
claimed yet. Base implementation is v69, source
`6d91d6430f88b5786ba905fc931ff7a9de164c2d`, including LoRA rematerialization
and the small 64 MiB bounded NF4 weight cache.

An own Triton gather kernel uses Triton's standard Philox primitive with the
same native CUDA thread, subsequence, counter, component and uniform-conversion
coordinates. It applies those bits to real rows and saves eight mask bits per
byte. It avoids allocating and writing full padded FP32 input/output and
boolean-mask buffers. It consumes the complete original padded counter range;
no random samples, training rows, steps or outputs are removed or rewound.

The isolated screen passed 37 bit-exact tests covering seeds, probabilities,
shapes, exceptional values, half output, gradients and following RNG draws.
The isolated GPU chain was 3.54-6.77x faster on three real-row counts; these
are not whole-training speedups. The batched gate/up alternative was rejected
because it was slower at the larger tested counts.

Reservation is qualified only for the exact private trainer consumer on the
main Python thread, Torch 2.11.0 / Triton 3.6.0 / T4, native tqdm monitor only,
no CUDA capture and the original large 1536-wide dropout path. Foreign worker
threads, public consumers and unsupported cases retain native dropout without
advancing RNG in the new path. This is not a general concurrent generator API.

New qualification also compares both LoRA branches' exact outputs, parameter
and residual gradients, retained graphs, masks and generator progression in
the actual integrated path. All 15 original protected files, fixed model,
dataset, task, optimizer, 64 steps, native CLI, head chunk and measurements
must stay unchanged. Selection requires original 10/10, every solution peak
<=7151 MiB and nonincreasing reported eval loss. Target remains <=25 seconds
for the whole command and is not claimed before measurement.

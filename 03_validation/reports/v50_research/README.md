# v50 native batched MLP screen

Inactive research, rejected before integration. This is not a full-task result.

12 of 14 shape/seed cases changed forward or backward FP16 values. At 1536 tokens both seeds were exact, but prepacked batched forward was 4.88–4.96 ms versus separate 4.29–4.32 ms; backward was 4.38–4.41 ms versus 3.87–3.91 ms. Stack-inclusive variants were slower still.

Keep selected v45. These timings do not establish a whole-command speedup.

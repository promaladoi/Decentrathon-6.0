# v66: fill the remaining frozen-weight cache and use larger head GEMMs

v65 completed the original 10/10 checks at 43.77 s / 6659 MiB. Its reported
loss increased by 0.00001, so it was not selected under the additional user
nonincrease policy. The unchanged organizer tolerances and this stricter
user policy remain separate; no waiver or threshold change is applied.

This candidate raises the process-local resident-weight budget from 2.4 to
2.8 GB. The fixed model's eligible FP16 projections total approximately
2.620 GB; this covers the last matrices still repeatedly decoded in v65.
Each measured subprocess still starts with an empty cache and pays initial
decode/allocation inside its own timer. All v65 mutation and stream guards
remain.

The internal frozen LM-head row chunk rises from 512 to 1024. This combines
native GEMMs when a batch has more supervised rows, while preserving every
row's logits, CE, entropy, argmax, actual-scale gradient and all Soup metrics.
This is not a change to the fixed training sequence length of 512 or batch
size of four. Existing head/gradient/retained-graph tests stay unchanged and
must pass before the original full paired benchmark runs.

Larger temporaries and resident weights deliberately trade memory for time.
Every measured solution run must remain <=7151 MiB, reported eval loss must
not increase, and original 10/10 must pass. The whole-command target is <=25
seconds. No new elapsed-time or memory success is claimed before measurement.

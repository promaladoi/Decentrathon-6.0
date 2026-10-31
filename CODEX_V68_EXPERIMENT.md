# v68: combine bounded full residency and guarded startup, keep head chunk 512

This candidate combines two independent time optimizations: v65's validated
internal frozen-weight reuse (with the 2.8 GB budget covering this fixed
model), and v67's exact-source-guarded native train CLI startup hook. The
public decoder remains fresh; all mutation/stream/budget guards remain.
No model, compiler cache or adapter is carried between measured processes.

The internal head chunk stays at v45's **512**. v66's larger 1024-row chunk
passed all tests but its observed training-loop median was 27.33 s versus
v65's 24.88 s. Its whole-command median was also slower, 46.20 versus 43.77 s.
Those are separately paired measurements, not a universal algorithm ranking.
This candidate therefore preserves the smaller chunk while testing the
remaining full-residency and startup combination.

All training arithmetic and the native train function/root callback remain.
The native small-M NF4 dispatch, state-dtype rounding, 64 steps, batch four,
sequence 512, data order, seeded dropout, optimizer and all Soup metrics stay
unchanged. The original 15 organizer files must remain byte-identical.

This is an unmeasured combination, not a speed or quality claim. The target
is <=25 seconds for the full command, every peak <=7151 MiB, nonincreasing
reported evaluation loss and original 10/10. Full original paired validation
is required before any selection decision.

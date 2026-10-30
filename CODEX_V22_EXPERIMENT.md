# v22: defer unrelated CLI command imports (T4 pending)

The user target is the complete unchanged command in 20–30 seconds, with the
same training task/quality and all ten checks. v17r1 has now completed at
84.28 / 54.19 seconds, 10/10, loss -0.002%; its archive still awaits collection.
The target has not been reached.

On top of v21, this revision avoids eagerly importing and registering 84 other
Soup command bindings when the requested command is `train`. It retains the
original train handler and all 13 CLI function/class definitions, including the
root callback, run(), parsing, error, audit and telemetry helpers, unchanged.
The same task, loader, optimizer, data, trainer and output path still execute.
No work is moved outside the timed process and no cached adapter/model is used.

This is a single-use import hook for `soup_cli.cli`, guarded by the exact
normalized source SHA-256 of the installed soup-cli 0.77.0 module. It transforms
only module imports and unrelated command registrations, verifies all function
ASTs are identical, and declines the optimization if the source is different or
a retained function refers to a deferred binding. Already imported CLI modules
and other requested commands follow the complete ordinary CLI.

Seven local CPU tests pass using the actual installed soup-cli 0.77.0 parser.
The ordinary and transformed `soup train --help` output/exit status are identical;
an invalid config follows the same parser/error behavior. Source-change fallback
and other-command behavior are also tested. This is startup-contract evidence,
not GPU training equivalence or a measured contest speedup.

The full solution suite has 181 cases (174 prior + 7 CLI). The runner also
preserves each base/solution train.log and metrics.json after measurement for
startup diagnosis. Organizer harness/task/tests are unchanged. Full T4 tests,
paired measurement, quality and all ten rules must pass before admission.

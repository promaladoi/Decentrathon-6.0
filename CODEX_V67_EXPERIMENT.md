# v67: reduce native train-command startup work on exact v45 math

This candidate returns to exact selected v45 training modules, then adds only
the team's previously qualified, exact-source-guarded train CLI import hook
from `7a8b38f`. There is no resident NF4 cache, LoRA rematerialization or
larger head chunk. Model arithmetic, data, seed, optimizer, dropout draws,
training steps, native training function and native root callback remain.

For only `soup train` and only the pinned Soup 0.77.0 CLI source hash, the hook
defers imports/registrations of 84 unrelated CLI command bindings. Every
retained original function definition is verified unchanged before compiling
the guarded module. Unsupported source revisions, already-loaded CLI modules
and other commands use the ordinary CLI. The original loader's source remains
available for inspection. Nothing is moved outside the measured subprocess.

The restored seven CPU tests compare original definitions, native `train
--help`, the real missing-config error and unsupported-command/source
fallbacks. They passed again locally against installed Soup 0.77.0. Full
integration tests and the original paired T4 harness are still required;
this is not a new elapsed-time or quality result.

The target is <=25 seconds for the whole command, not only the training loop.
Every solution peak must stay <=7151 MiB; reported eval loss must not increase
and all ten original checks must pass. The public selected implementation
stays v45 until measured comparison justifies replacement.

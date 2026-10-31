# Instrumented v45 startup timeline

Python profile callbacks cause substantial observer overhead: the outer run is 82.42 s and the measured training loop is 37.91 s, compared with the unmodified admitted v45 result 45.03 s / 26.54 s. Do not compare these as candidate performance, or sum overlapping inclusive intervals.

The trace identifies native import/model-loading/trainer stages and shows adapter saving is a small part of this observed run. It does not establish how many normal-run seconds an import optimization could save.

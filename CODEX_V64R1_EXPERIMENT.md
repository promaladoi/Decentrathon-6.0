# v64r1: correct the overridden-Linear qualification fixture

The original v64 archive stopped before model training: 10 targeted tests
passed and the overridden-Linear fixture failed. Its closure captured the
name `original`, which was subsequently rebound to the rematerialization
function. The fixture then called that unrelated function with one argument.

This revision gives the native Linear forward and rematerialization function
separate names. The failing observer/fallback assertion is retained. No
production Python module under `solution/` changes from measured-source
v64 (`155972c13b59f8955a2eb8dd7651b0f2cc5c8b9f`); only the test changes.
Original organizer files and all ten original criteria remain untouched.

The additional user report now uses the latest 25-second whole-command
target. Historical version reports keep their original targets. The user
memory cap remains 7151 MiB and reported evaluation loss must not increase.

New results are written as `codex_v64r1.zip`, preserving the failed v64
archive. The exact v45 public implementation stays selected until verified
results justify any replacement. No new training time, memory or quality
result is claimed yet.

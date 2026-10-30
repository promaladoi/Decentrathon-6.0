# v41: isolate the new CPU work on the admitted v34 active path

Experimental control, not a measured improvement.

Restores exactly v34's three active integration files (`solution/__init__.py`,
`solution/team_vladimir/mlp.py`, `solution/team_vladimir/unpad.py`) while retaining
the new byte-identical tokenizer canonicalization, equivalent cache-hit guards,
and prompt release of 512-row vocabulary temporaries.

Merei's helper implementations and **all their tests remain** in `solution/`.
They are inactive in this control, rather than silently combining their effect
with the newer CPU changes. Team credit and v36--v40 evidence are preserved.

The full original test suite and all ten unchanged organizer checks must pass.
Every measured solution run must fit 7151 MiB, evaluation loss must not exceed its
reported paired base, and 30 seconds remains an unmet target until measured.

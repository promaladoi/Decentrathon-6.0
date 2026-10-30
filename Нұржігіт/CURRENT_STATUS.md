# Current status after participant supplied v13 results

- **v13 completed:** code `13eaa8a025752db94a020102fa3ecd90951a2fec`.
  The participant supplied full console output with 106 kernel tests, 79 Soup tests,
  all 10 checks, speedup 1.398x, 5761 MiB and eval loss 2.04966. Own share 11.3%.
  Raw JSON, source archive and adapter still need to be downloaded/hash-verified.
- **v12 fully archived fallback:** code `bd2b49a`, speedup 1.1608x, 7385 MiB,
  64 kernel tests, 79 Soup tests, 10/10 checks, raw hashes/source verified locally.
- **v14 NOT confirmed launched:** code `fd5e4826645b85106859cae7a254297e52597fbf`.
  Browser control disconnected. The participant says no v14 cell is present.
  A fresh pinned `templates/codex_v14.ipynb` is prepared. It first tests exact
  dropout outputs, gradients and RNG state, then all 110 kernel tests, 79 Soup
  tests and two paired benchmark runs. No v14 performance or quality is claimed.
- The fresh notebook uses a new unique checkout, public mirror, and no GitHub
  token. Existing runs are preserved. It can also export existing v13 artifacts.
- A public saved final run remains necessary after the team selects a submission.
- No 70–90% improvement is claimed. Vladimir's user-reported 1.468x / 2.543x result
  is currently stronger than the reported v13 ratios; use a controlled comparison.

# Current optimization status

- **Accepted fallback:** v12, code `bd2b49ac6914a3b8aa30133b615c1a23ddf34338`.
  Two paired T4 runs, speedup 1.1608x, 7385 MiB peak, eval loss 2.04963,
  all 10 checks, 64 kernel tests and 79 Soup tests passed.
- **Combined v13:** code `13eaa8a025752db94a020102fa3ecd90951a2fec`.
  Credits Vladimir's unpadding/model path and Nurzhigit/Codex's parallel head.
  106 kernel tests + 79 Soup tests passed. Visible base times 89.06 / 87.97 s;
  solution 62.93 / 63.68 s, 5761 MiB. Final eval/check report not yet collected.
- **Combined v14:** initial code `322d88e`, then `fd5e4826645b85106859cae7a254297e52597fbf`
  removes token-count specialization so different lengths reuse the compiled
  dropout kernel. A queued notebook cell was updated to the newer SHA; the actual
  executed revision must be read from its output/archive before accepting results.
- The CUA browser connection closed during v13 evaluation. GPU processes may
  continue. Do not blindly rerun or change the measured checkout; first inspect
  the existing cell/output and GPU process state. Reports are under
  `/kaggle/working/nurzhigit_v13/runs_codex_v13` and the corresponding v14 path.
- Interactive Kaggle draft still has a save conflict. Downloaded artifacts and
  pinned notebooks are stored separately; a public saved final run is still needed
  after the team selects its submission.
- No 70–90% gain is claimed. `speedup` is throughput ratio; runtime reduction is
  `1 - solution_time/base_time`. Other teammates' reports must be independently
  checked before deciding which final team version to submit.

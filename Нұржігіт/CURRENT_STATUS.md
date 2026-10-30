# Current status

- Accepted console result: v13 `13eaa8a`, 1.398x, 5761 MiB, eval 2.04966,
  106 kernel tests + 79 Soup tests, all 10 checks. Raw v13 archive pending.
- Fully archived fallback: v12 `bd2b49a`, 1.1608x, 7385 MiB, all 10 checks;
  exact raw-report and source hashes verified locally.
- v14 `fd5e482`: 4 tests failed; no training. v15 `389f037`: 6 tests failed;
  no training. v15 archive downloaded; 31 source files match Git exactly.
- Actual issue: strided row-index views were read as dense indices. The earlier
  claim that RNG mapping differed was not established. v16 normalizes row storage
  once and saves it for backward. All strict numerical/RNG tests remain.
- v16 source: `4a5cfd20fd79e597e15919f3f19f959a6b4d689a`. It retains startup compilation, adds 2 layout
  regression cases (8 dropout / 115 total kernel tests), and improves failed
  outcome metadata. GPU tests, quality and performance are NOT yet verified.
- Use `templates/run_v16_cell.py` after the failed v15 cell. It preserves the
  existing data and prior run outputs, selects the exact commit and runs the
  guarded validator with a new `runs_codex_v16` output directory.
- Browser automation remains disconnected/blocked. No autonomous v16 run is claimed.

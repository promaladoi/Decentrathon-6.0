"""Stage timestamps in the training log, to see where the measured time of `soup train` goes.

A solution run spends about 24 s outside the 64 training steps (Python and library start-up,
data, the 4-bit model load, LoRA, saving), as much as the steps themselves. These lines show how
that splits, so the next optimization targets the right stage. One line per stage goes to stderr,
which the harness writes to train.log, with the process age in seconds:

    solution: t=  5.1 s  apply()
    solution: t= 12.4 s  Soup setup start
    ...

It only observes: the wrappers pass arguments, results and exceptions through unchanged.
"""

import atexit
import functools
import os
import sys
import time

_START = time.perf_counter()


def _age():
    """Seconds since this process started (Linux /proc); else since this module loaded."""
    try:
        with open("/proc/self/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")  # field 22: start time
        with open("/proc/uptime") as handle:
            return float(handle.read().split()[0]) - started
    except Exception:
        return time.perf_counter() - _START


def log(stage):
    print(f"solution: t={_age():6.1f} s  {stage}", file=sys.stderr, flush=True)


def _wrap(cls, name, label):
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_solution_timeline", False):
        return

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        log(f"{label} start")
        try:
            return original(*args, **kwargs)
        finally:
            log(f"{label} end")

    wrapper._solution_timeline = True
    setattr(cls, name, wrapper)


def install():
    log("apply()")
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
    except Exception:
        return
    _wrap(SFTTrainerWrapper, "setup", "Soup setup (data, model, LoRA, trainer)")
    _wrap(SFTTrainerWrapper, "_setup_transformers", "model load + LoRA")
    _wrap(SFTTrainerWrapper, "train", "Soup train (steps + save)")
    atexit.register(log, "process exit")

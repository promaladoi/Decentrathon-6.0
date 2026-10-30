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
    """Anais: portable relative time; no reads outside the repository/task data."""
    return time.perf_counter() - _START


_ENABLED = False
_SEEN = set()


def log(stage):
    # leading newline: the training progress bar may be mid-line on stderr
    print(f"\nsolution: t={_age():6.1f} s  {stage}", file=sys.stderr, flush=True)


def once(stage, sync=False):
    """Log `stage` the first time only (step 1 marks); `sync` waits for the GPU first."""
    if not _ENABLED or stage in _SEEN:
        return
    _SEEN.add(stage)
    if sync:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    log(stage)


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


def _wrap_first_step():
    """Mark the end of step 1's forward + backward (Trainer.training_step, before the
    optimizer), so step 1's time splits into forward and backward."""
    import transformers

    original = transformers.Trainer.training_step
    if getattr(original, "_solution_timeline", False):
        return

    @functools.wraps(original)
    def training_step(*args, **kwargs):
        result = original(*args, **kwargs)
        once("step 1 forward + backward end", sync=True)
        return result

    training_step._solution_timeline = True
    transformers.Trainer.training_step = training_step


def install():
    global _ENABLED
    _ENABLED = True
    log("apply()")
    try:
        _wrap_first_step()
    except Exception:
        pass
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
    except Exception:
        return
    _wrap(SFTTrainerWrapper, "setup", "Soup setup (data, model, LoRA, trainer)")
    _wrap(SFTTrainerWrapper, "_setup_transformers", "model load + LoRA")
    _wrap(SFTTrainerWrapper, "train", "Soup train (steps + save)")
    atexit.register(log, "process exit")

"""Compile our Triton kernels in a background thread while Soup starts.

Triton compiles a kernel at its first launch, once per signature, and that time is part of the
measured `soup train` command. apply() runs before Soup imports its trainer, reads the data and
builds the 4-bit model, which takes many seconds; this thread compiles the signatures the training
run will launch during that time (kernel.warmup compiles without launching anything), in the order
the first step needs them. The first step waits for the thread instead of compiling a kernel a
second time. It is the same compilation the first step would do, only earlier: nothing is read
from or kept for another run, and a signature that does not match the run only costs a compile at
first launch, as without warm-up. The harness counts warm-up calls as launches but does not time
them. Two lines in the training log say when the thread finished and how long step 1 waited.
"""

import sys
import threading
import time

_state = {"thread": None, "started": None}


def _steps(unpad):
    from . import lmhead, mlp, nf4, rmsnorm, rope
    from . import unpad as fast

    if unpad:  # the unpadded training step, in order of first use
        return [nf4.warmup,
                lambda: rmsnorm.warmup(parts=("add_fwd",)),
                lambda: fast.warmup(parts=("fwd",)),
                lambda: mlp.warmup(parts=("fwd",)),
                lmhead.warmup,
                lambda: rmsnorm.warmup(parts=("add_bwd",)),
                lambda: mlp.warmup(parts=("bwd",)),
                lambda: fast.warmup(parts=("bwd",))]
    return [nf4.warmup, rmsnorm.warmup, rope.warmup, mlp.warmup]


def _compile_all(unpad):
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(torch.cuda.current_device())
    failed = []
    for step in _steps(unpad):
        try:
            step()
        except Exception as exc:  # a failed warm-up only means the first launch compiles
            failed.append(repr(exc))
    note = f"; failed: {failed}" if failed else ""
    print(f"solution: warm-up thread compiled the kernels by "
          f"{time.perf_counter() - _state['started']:.1f} s after apply(){note}",
          file=sys.stderr, flush=True)


def start(unpad=True):
    _state["started"] = time.perf_counter()
    thread = threading.Thread(target=_compile_all, args=(unpad,), name="solution-triton-warmup",
                              daemon=True)
    _state["thread"] = thread
    thread.start()
    return thread


def wait(timeout=60.0):
    """Before the first training step: let the thread finish rather than compile twice."""
    thread = _state["thread"]
    if thread is None:
        return
    _state["thread"] = None
    began = time.perf_counter()
    thread.join(timeout)
    print(f"solution: step 1 waited {time.perf_counter() - began:.2f} s for the warm-up thread "
          f"({began - _state['started']:.1f} s after apply())", file=sys.stderr, flush=True)

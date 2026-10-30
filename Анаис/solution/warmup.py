"""Compile our Triton kernels in a background thread while Soup starts.

Triton compiles a kernel at its first launch, once per signature, and that time is part of the
measured `soup train` command. apply() runs before Soup imports its trainer, reads the data and
builds the 4-bit model, which takes many seconds; this thread compiles the signatures the training
run will launch during that time (kernel.warmup compiles without launching anything), in the order
the first step needs them. The first step waits for the thread instead of compiling a kernel a
second time. After compiling, it also builds each kernel's launcher and loads it on the GPU,
the other one-time cost of a first launch (see _load_compiled). It is the same work the first
step would do, only earlier: nothing is read from or kept for another run, and a signature that
does not match the run only costs a compile at first launch, as without warm-up. The harness
counts warm-up calls as launches but does not time them. Two lines in the training log say when
the thread finished and how long step 1 waited.
"""

import sys
import threading
import time

_state = {"thread": None, "started": None}


def _steps(unpad):
    from . import lmhead, mlp, nf4, rmsnorm, rope
    from . import unpad as fast
    from . import dropout as compact_dropout

    if unpad:  # the unpadded training step, in order of first use
        steps = [nf4.warmup,
                lambda: rmsnorm.warmup(parts=("add_fwd",)),
                lambda: fast.warmup(parts=("fwd",)),
                lambda: mlp.warmup(parts=("fwd",)),
                lmhead.warmup,
                lambda: rmsnorm.warmup(parts=("add_bwd",)),
                lambda: mlp.warmup(parts=("bwd",)),
                lambda: fast.warmup(parts=("bwd",))]
        if fast.REAL_ROW_DROPOUT:
            # After QKV projection, before RoPEPad: compile arithmetic only.
            # This callback never invokes native_dropout or consumes RNG.
            steps.insert(2, compact_dropout.warmup)
        return steps
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
    compiled = time.perf_counter() - _state["started"]
    try:
        loaded = _load_compiled()
    except Exception as exc:  # the first launch then builds and loads them itself
        loaded = 0
        failed.append(repr(exc))
    note = f"; failed: {failed}" if failed else ""
    print(f"solution: warm-up thread compiled the kernels by {compiled:.1f} s and prepared "
          f"{loaded} launchers by {time.perf_counter() - _state['started']:.1f} s after "
          f"apply(){note}", file=sys.stderr, flush=True)


def _load_compiled():
    """Build the launcher and load the GPU module of every kernel compiled so far, as its first
    launch would. Triton 3.6 does both lazily at the first launch (CompiledKernel._init_handles):
    it compiles a small C launcher per signature with the system C compiler, which the
    harness's fresh per-run cache cannot reuse, and loads the kernel binary into the CUDA
    context. Measured, that was about 2.3 s of step 1. Done here, step 1 only launches."""
    from triton.runtime.jit import JITFunction

    from . import dropout, lmhead, mlp, nf4, rmsnorm, rope, swiglu, unpad

    count = 0
    for module in (nf4, rmsnorm, unpad, lmhead, dropout, swiglu, rope, mlp):
        for fn in list(vars(module).values()):
            if not isinstance(fn, JITFunction):
                continue
            for entry in list(getattr(fn, "device_caches", {}).values()):
                kernels = entry[0] if isinstance(entry, tuple) else entry
                for kernel in list(kernels.values()):
                    init = getattr(kernel, "_init_handles", None)
                    if init is not None:
                        init()
                        count += 1
    return count


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
    print(f"\nsolution: step 1 waited {time.perf_counter() - began:.2f} s for the warm-up thread "
          f"({began - _state['started']:.1f} s after apply())", file=sys.stderr, flush=True)

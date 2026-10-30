"""Compile our Triton kernels in a background thread while Soup starts.

Triton compiles a kernel at its first launch, once per signature, and that time is part of the
measured `soup train` command. apply() runs before Soup imports its trainer, reads the data and
builds the 4-bit model, which takes many seconds; this thread compiles the signatures the training
run will launch during that time (kernel.warmup compiles without launching anything). It is the
same compilation the first step would do, only earlier: nothing is read from or kept for another
run, and a signature that does not match the run only costs a compile at first launch, as without
warm-up. The harness counts warm-up calls as launches but does not time them.
"""

import threading


def _compile_all():
    import torch

    from . import mlp, nf4, rmsnorm, rope

    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(torch.cuda.current_device())
    for step in (nf4.warmup, rmsnorm.warmup, rope.warmup, mlp.warmup):
        try:
            step()
        except Exception:  # a failed warm-up only means the first launch compiles
            pass


def start():
    thread = threading.Thread(target=_compile_all, name="solution-triton-warmup", daemon=True)
    thread.start()
    return thread

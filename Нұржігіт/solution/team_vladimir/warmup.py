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


def _compile_all(unpad):
    import torch

    from . import lmhead, mlp, nf4, rmsnorm, rope
    from . import unpad as unpad_module

    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(torch.cuda.current_device())
    steps = [nf4.warmup, rmsnorm.warmup, mlp.warmup]
    # the unpadded forward does RoPE inside its own kernel; rope.py is its fallback
    steps += [unpad_module.warmup, lmhead.warmup] if unpad else [rope.warmup]
    for step in steps:
        try:
            step()
        except Exception:  # a failed warm-up only means the first launch compiles
            pass


def start(unpad=True):
    thread = threading.Thread(target=_compile_all, args=(unpad,), name="solution-triton-warmup",
                              daemon=True)
    thread.start()
    return thread

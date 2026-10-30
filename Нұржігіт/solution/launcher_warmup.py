"""Prepare compiled launchers inside the existing measured startup thread.

Vladimir's 13d3770 identified Triton's lazy C-launcher/module initialization as
a first-step cost. This adapts that team contribution to include our registered
kernels too, without importing inactive files or duplicating shared functions.
"""
import sys


def prepare(modules, jit_type):
    seen_functions, seen_kernels = set(), set()
    count = 0
    for module in list(modules):
        for fn in list(vars(module).values()):
            if not isinstance(fn, jit_type) or id(fn) in seen_functions:
                continue
            seen_functions.add(id(fn))
            for entry in list(getattr(fn, 'device_caches', {}).values()):
                kernels = entry[0] if isinstance(entry, tuple) else entry
                if not isinstance(kernels, dict):
                    continue
                for kernel in list(kernels.values()):
                    if id(kernel) in seen_kernels:
                        continue
                    seen_kernels.add(id(kernel))
                    init = getattr(kernel, '_init_handles', None)
                    if init is not None:
                        init()
                        count += 1
    return count


def load_compiled():
    from triton.runtime.jit import JITFunction
    package=__package__.split('.')[0]
    modules=[m for n,m in list(sys.modules.items()) if m is not None and
             (n==package or n.startswith(package+'.'))]
    count=prepare(modules,JITFunction)
    print(f'solution: prepared {count} in-process compiled kernel launchers',
          file=sys.stderr,flush=True)


def patch_launcher_warmup():
    from .team_vladimir import warmup
    original=warmup._steps
    if getattr(original,'_solution_load_launchers',False):
        return
    def steps(unpad):
        return original(unpad)+[load_compiled]
    steps._solution_load_launchers=True
    warmup._steps=steps

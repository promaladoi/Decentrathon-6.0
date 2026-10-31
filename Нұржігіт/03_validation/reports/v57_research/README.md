# v57 original shared-MMA screen

Status: `research_rejected`. Source `ec0c5b822dc4e6f67207c1335984b24d28d1140f`; both research source files verified byte for byte against Git. This is not model training or a task score. Production v45 is unchanged.

Error: `CompilationError: at 12:13:
def shared_turing_mma_kernel(X, W, Y, M, N, K,
                              BM: tl.constexpr, GROUP_M: tl.constexpr,
                              ASM: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, 64)
    per_group = GROUP_M * nn
    group = pid // per_group
    first_m = group * GROUP_M
    active_m = tl.minimum(nm-first_m, GROUP_M)
    pid_m = first_m+(pid % per_group) % active_m
    pid_n = (pid % per_group) // active_m
    unused = tl.inline_asm_elementwise(ASM,
             ^
invalid operands of type pointer<fp32> and pointer<fp16>`

See screen.json for unaltered numerical and timing evidence. No original ten-check admission or <=27-second claim is made for this isolated screen.

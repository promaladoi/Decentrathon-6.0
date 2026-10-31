# v63: exact v45 arithmetic, measured NF4 launch choices and shorter lifetimes

Starting solution is selected v45 (`7a3db8fd5e24bbcb58affb475007a5cf93c56f07`).
The v48 unpad.py lifetime-only change releases dead references earlier; its
original paired report previously admitted 4115 MiB. No saved autograd state
is removed or recomputed. All other v45 runtime modules remain unchanged.

The v62 native-exact geometry screen motivates a small NF4 dispatch change:
on SM75, FP16/nested NF4 block64 matrices with at least 8M elements use 1024
elements / 8 warps; 2M-8M use 2048 / 4. Remaining inputs keep v45's 1024 / 4.
The elementwise arithmetic, casts, no-FMA option, native GEMM choice, model,
tokens, dropout RNG, optimizer, metrics and original harness stay intact.
The selector is cached by plain hardware/geometry metadata only.

Measured decoder-only gains were about 5.6% and 5.8%, not whole-command gains.
No full-task result is claimed before paired native measurement. Sixteen new
GPU tests require bitwise agreement for large matrices, FP16/FP32 weights and
outputs, nested/non-nested scales. Existing team/Soup tests and original ten
checks remain required, alongside <=7151 MiB and nonincreasing reported loss.

CUDA graphs and the shared-MMA research are not installed in training.
Selected public v45 remains unchanged pending a justified result. The whole
command <=27-second target has not been achieved by the isolated screens.

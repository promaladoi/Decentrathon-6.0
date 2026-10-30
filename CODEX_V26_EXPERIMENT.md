# v26: prepare launchers in the measured startup thread

Experimental on top of v25; all numeric operations unchanged. Vladimir's
collaboration commit `13d377043fa4cdad121973a3f6179d546918e9f4` identified lazy
Triton `_init_handles` costs in step 1 and moved those into the existing compiler
thread. This contribution is credited to Vladimir and his tools.

Our integration prepares all already-compiled JIT functions in loaded solution
modules, including Nurzhigit's kernels, deduplicates aliases, and falls back to
normal first-launch initialization if preparation fails. It calls no kernels
and uses no cache from another run. Compilation, C-launcher construction and
CUDA-module loading stay inside the timed `soup train` process. The unchanged
real own-kernel timing still has to meet the organizer's 5% rule.

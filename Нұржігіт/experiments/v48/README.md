# v48: v45 arithmetic with shorter temporary lifetimes only

Compared with verified v45 source 7a3db8fd5e24bbcb58affb475007a5cf93c56f07,
the production solution differs by five added lines in unpad.py: four `del`
statements and one explanatory comment. All other solution files are exact
v45 Git bytes. No kernel math, shape, dtype, RNG, warmup or module call changes.

Release projection outputs after custom RoPE creates its outputs, the old MLP
output after residual normalization, attention output after its residual
normalization, and normalization output after MLP. Autograd's own saved tensors
retain their original lifetime; these statements remove only Python locals.

v46 passed original 10/10 but was not selected: 45.75 s, 4135 MiB and reported
eval 2.04967 versus paired base 2.04963. That fails the user's stricter
nonincreasing-quality requirement. v47 inherited that dropout cast and was
superseded before measurement. v48 restores v45 dropout, warmup and LoRA math.

No performance/quality result is claimed for v48 until full paired T4 runs and
the original ten checks finish. v45 remains selected meanwhile. The target is
<=30 s whole command, <=7151 MiB and nonincreasing reported eval loss.

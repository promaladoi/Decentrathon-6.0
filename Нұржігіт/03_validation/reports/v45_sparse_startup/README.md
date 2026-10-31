# Sparse startup timestamps

An in-memory diagnostic AST adds timestamps around main-body statement boundaries. The original harness file, native Soup/Transformers callees and selected source are intact. This is not a contest score.

Outer diagnostic time: 49.55 s; loop: 28.75 s. Before the native CLI: about 11.18 s, including torch import 1.64 s, CUDA initialization 0.38 s, Transformers import 3.20 s and solution application 5.59 s. CLI time includes model/setup, training and saving. These stage measurements do not prove a speedup or a hardware lower bound.

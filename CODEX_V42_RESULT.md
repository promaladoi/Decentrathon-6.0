# v42 screen: rejected before numerical sweep and full training

Participant-supplied output of the pinned Kaggle screen at research revision
`aea2dbb8076c0c3a1e1d4c6ea361abf2df5b01e7` reports Tesla T4, Torch
2.11.0+cu128 and Triton 3.6.0. Its generated gate/up projection PTX contained
no Tensor Core MMA. The bounded screen rejected this candidate with exit 2.
The 15 numerical research tests, microbenchmark sweep and full-model training
were therefore **not run**. There is no v42 speed, memory or quality claim.
Production was not changed by this screen; the selected snapshot remains v34.

The same screen verified the original v41 ZIP in the Kaggle VM against all 53
Python source files of `db4908e8b486874fda8c6eacc230166116ccad01` and the
original check/result/source hashes. Reported v41 archive SHA256:
`a454cfe36a816bfbe616cb4b5fb6fd297cbe01a25b529124e582da611c7ad5ee`.
Reported solution SHA256:
`aaab004b2b594ccb0f664f44c1c7533cf365f8b05c7af761bbfb58dd305983e5`.

The participant reports two paired runs per side: base 80.72 s / 14345 MiB /
loss 2.04967; solution 44.61 s / 7011 MiB / loss 2.04965. All ten original
checks passed and every solution peak fits 7151 MiB. The training loop is
24.55 s, while full-command time is 44.61 s. The 30-second target is unmet.

Evidence level: console supplied by the participant from the pinned screen.
The original ZIPs, PTX and timing reports have not been collected locally.
No original result.json, checks.json or collection_proof.json is reconstructed
from this console. The screen archive remains in the user's Kaggle session at
`/kaggle/working/codex_v42_screen_2fcffcf1.zip`.

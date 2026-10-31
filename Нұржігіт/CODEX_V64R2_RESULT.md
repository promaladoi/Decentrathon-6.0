# v64r2 verified original outcome

Exact source: `5c36601c8302da8e0910b7697637fea4b3a0393d`; fifty Python files verified against Git byte for byte.

The failed v64 fixture and v64r1 legacy-helper qualification are preserved. v64r2 retains all eleven rematerialization tests and all original scaling assertions; its early qualification runs both modules.

| Side | Median whole command | NVML peak | Reported eval loss |
|---|---:|---:|---:|
| Base | 86.33 s | 14345 MiB | 2.04966 |
| v64r2 | 47.61 s | 3915 MiB | 2.04963 |

Original checks: **10/10**. Two base and two solution runs. Training-loop time: 28.36 s, separate from whole-command time.

The <=25-second whole-command target is not met.

User memory cap <=7151 MiB: pass. Nonincreasing reported evaluation loss: pass.

LoRA-A backward inputs are reconstructed with the same arithmetic from shared RMS residuals, inverse RMS, frozen weights and native dropout mask bits. No GEMM or random draw is added.

Current public selection stays measured v45 until an explicit evidence-backed comparison/selection decision.

[Source/result proof](03_validation/reports/codex_v64r2/collection_proof.json). [User limits](03_validation/reports/codex_v64r2/user_constraints.json). [Candidate description](CODEX_V64R2_EXPERIMENT.md).

[Completed same-session comparison and selection decision](CODEX_V64_COMPARISON.md).

[Four original ZIPs and checksum manifest](https://github.com/promaladoi/Decentrathon-6.0/releases/tag/nurzhigit-evidence-through-v64r2-20261031): failed v64, failed v64r1, admitted v64r2 and the exact same-session v45 control. Five release assets have verified server SHA-256 digests and public download links. [Publication receipt](04_submission/v64_revisions_publication.json). [Protected-file/source audit](04_submission/v64_revisions_source_audit.json). [Selection and source verification](04_submission/v64_revisions_selection.json).

"""Optional synthetic microbenchmarks, separate from the official training score.

Run: python -m solution.benchmark --out microbench.json
Only a Kaggle T4 may be used for challenge measurements. This script refuses
other GPUs and never changes device visibility or allocator configuration.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics

import torch


def measure(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    start_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times),
            "peak_extra_allocated_mib":
                (torch.cuda.max_memory_allocated() - start_bytes) / 2**20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("microbench.json"))
    args = parser.parse_args()
    if min(args.rows, args.vocab, args.repeats) < 1:
        parser.error("rows, vocab and repeats must be positive")
    if not torch.cuda.is_available():
        parser.exit(2, "Kaggle T4 CUDA runtime required; no benchmark was run.\n")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        parser.exit(2, f"Kaggle T4 required; detected {name}; no benchmark was run.\n")
    from .rmsnorm import frozen_rmsnorm
    from .vocab import cross_entropy, entropy_from_logits

    torch.manual_seed(17)
    logits = torch.randn((args.rows, args.vocab), device="cuda", requires_grad=True)
    labels = torch.randint(args.vocab, (args.rows,), device="cuda")
    labels[::4] = -100

    def ce_base():
        torch.autograd.grad(torch.nn.functional.cross_entropy(logits, labels), logits)

    def ce_ours():
        torch.autograd.grad(cross_entropy(logits, labels), logits)

    @torch.no_grad()
    def entropy_base():
        return logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)

    @torch.no_grad()
    def entropy_ours():
        return entropy_from_logits(logits)

    x = torch.randn((args.rows, 1536), device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda")
    dy = torch.randn_like(x)

    def rms_base():
        xf = x.float()
        out = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        torch.autograd.grad(out, x, dy)

    def rms_ours():
        torch.autograd.grad(frozen_rmsnorm(x, weight), x, dy)

    results = {"kind": "synthetic_microbenchmark_not_training_score",
               "device": name, "torch": torch.__version__, "rows": args.rows,
               "vocabulary": args.vocab, "repeats": args.repeats,
               "timing": "CUDA events after 3 warmups; excludes compilation",
               "memory": "extra PyTorch allocated peak, not official nvidia-smi peak"}
    for operation, base, ours in (("cross_entropy_forward_backward", ce_base, ce_ours),
                                  ("entropy", entropy_base, entropy_ours),
                                  ("rmsnorm_forward_backward", rms_base, rms_ours)):
        results[operation] = {"base": measure(base, args.repeats),
                              "solution": measure(ours, args.repeats)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

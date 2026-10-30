"""Explicit T4-only NF4 GEMM experiment; never installs a training dispatch.

Run from the task root: python -m solution.benchmark_nf4_gemm --out nf4_gemm.json
Use a fresh caller-provided TRITON_CACHE_DIR for genuine cold compilation figures.
The report separates first-call wall time, warmed latency, numerical error and
extra allocated tensor memory. These are not official full-training metrics.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton

from .nf4_gemm import BENCHMARK_TILES, fused_nf4_gemm, reference_nf4_gemm


def _latency(function, repetitions):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repetitions


def _extra_memory(function):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = function()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    output_bytes = result.numel() * result.element_size()
    del result
    return {"extra_peak_bytes": extra, "output_bytes": output_bytes}


@torch.no_grad()
def run(rows, repetitions, limit_tiles):
    import bitsandbytes.functional as bnbf

    name = torch.cuda.get_device_name()
    if "T4" not in name:
        raise RuntimeError("This performance experiment must run on the contest T4")
    report = {
        "device": name, "torch": torch.__version__, "triton": triton.__version__,
        "timing": "first invocation synchronized wall; warm median of three alternating-order rounds",
        "first_call_warning": "includes compilation only if signature is absent from caller's cache",
        "cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "reference": "current decoder + torch.mm, no retained dense weight",
        "memory": "extra torch allocated peak including output, not NVML/full-training peak",
        "training_dispatch_installed": False, "whole_training_gain_claimed": False,
        "rows": rows, "repetitions": repetitions, "sources": {}, "cases": []}
    for filename in ("nf4_gemm.py", "benchmark_nf4_gemm.py", "nf4.py"):
        path = Path(__file__).with_name(filename)
        report["sources"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.manual_seed(773)
    for shape in ((8960, 1536), (1536, 8960)):
        dense = torch.randn(shape, device="cuda", dtype=torch.float16) * .02
        packed, state = bnbf.quantize_4bit(dense, blocksize=64,
                                           compress_statistics=True, quant_type="nf4")
        del dense
        for transpose in (True, False):
            reduction = shape[1 if transpose else 0]
            for m in rows:
                a = torch.randn((m, reduction), device="cuda", dtype=torch.float16) * .2
                reference = lambda: reference_nf4_gemm(a, packed, state, transpose_weight=transpose)
                expected = reference()
                torch.cuda.synchronize()
                for tile in BENCHMARK_TILES[:limit_tiles]:
                    candidate = lambda: fused_nf4_gemm(a, packed, state,
                        transpose_weight=transpose, tile=tile)
                    case = {"weight_shape": list(shape), "m": m,
                            "direction": "forward" if transpose else "input_gradient",
                            "tile": asdict(tile)}
                    try:
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        actual = candidate()
                        torch.cuda.synchronize()
                        case["first_call_wall_s"] = time.perf_counter() - began
                        delta = actual.float() - expected.float()
                        case["max_abs_error"] = float(delta.abs().max())
                        case["relative_l2_error"] = float(delta.norm() / expected.float().norm().clamp_min(1e-8))
                        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
                        if case["relative_l2_error"] >= 2e-3:
                            raise AssertionError("relative L2 error exceeds experimental gate")
                        case["numerics_passed"] = True
                        del actual, delta
                        rounds = []
                        for index in range(3):
                            order = (reference, candidate) if index % 2 == 0 else (candidate, reference)
                            timing = [_latency(fn, repetitions) for fn in order]
                            base_ms, fused_ms = timing if index % 2 == 0 else timing[::-1]
                            rounds.append({"reference_ms": base_ms, "fused_ms": fused_ms})
                        case["rounds"] = rounds
                        base_ms = statistics.median(item["reference_ms"] for item in rounds)
                        fused_ms = statistics.median(item["fused_ms"] for item in rounds)
                        case.update(reference_ms=base_ms, fused_ms=fused_ms, micro_speedup=base_ms / fused_ms)
                        # The parity output is part of the identical before
                        # allocation baseline, subtracted for both sides.
                        case["reference_memory"] = _extra_memory(reference)
                        case["fused_memory"] = _extra_memory(candidate)
                        case["ok"] = True
                    except Exception as exc:
                        case.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                del expected, a
        del packed, state
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 769, 1536])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--tiles", type=int, default=len(BENCHMARK_TILES))
    parser.add_argument("--out", type=Path, default=Path("nf4_gemm_benchmark.json"))
    args = parser.parse_args()
    if any(m <= 0 for m in args.rows) or args.repetitions < 1 or not 1 <= args.tiles <= len(BENCHMARK_TILES):
        parser.error("positive rows/repetitions and a valid tile count are required")
    report = run(args.rows, args.repetitions, args.tiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.out}", flush=True)


if __name__ == "__main__":
    main()

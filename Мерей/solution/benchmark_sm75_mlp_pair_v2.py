"""Inactive SM75 pair V2 diagnostic, including ephemeral scale predecode.

Run: python -m solution.benchmark_sm75_mlp_pair_v2 --out /kaggle/working/pair_v2.json
Use a fresh caller-owned TRITON_CACHE_DIR for genuine cold first-call times.
The existing strict native partial-ULP gate is unchanged. This is not a
whole-training score, and research timings never admit an accuracy failure.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch
import triton

from .benchmark_mlp_grad_pair import (
    accuracy_metrics, reference_parts, reference_output, _memory, _sample, _summary)
from .nf4_scale_pair import predecode_nested_scales_pair
from .sm75_mlp_pair_v2 import CONFIGS, DEFAULT_CONFIG, sm75_mlp_grad_pair_v2


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[833])
    parser.add_argument("--configs", nargs="+", default=[DEFAULT_CONFIG])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--time-unqualified", action="store_true")
    args = parser.parse_args()
    configs = list(CONFIGS) if args.configs == ["all"] else args.configs
    if any(value not in CONFIGS for value in configs) or min(args.rows) <= 0 \
            or args.rounds < 1 or args.samples < 1:
        parser.error("positive rows/rounds/samples and valid configs required")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("this diagnostic requires the task Tesla T4")
    import bitsandbytes.functional as bnb

    report = {
        "kind": "inactive_sm75_pair_v2_ephemeral_scales_not_training_score",
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "real_turing_mma_required": True, "dense_weight_cache": False, "split_k_workspace": False,
        "persistent_scale_cache": False, "scale_workspace_bytes": 2 * (8960 * 1536 // 64) * 4,
        "candidate_timing_includes_scale_predecode_each_call": True,
        "reference": "existing on-demand NF4 decode + two native half GEMMs + FP32 sum",
        "cold_warning": "first call includes compilation only if caller cache lacks both kernel signatures",
        "numerical_gate": "unchanged benchmark_mlp_grad_pair.accuracy_metrics",
        "rounds": args.rounds, "samples_per_block": args.samples,
        "whole_training_gain_claimed": False, "sources": {}, "cases": []}
    for filename in ("sm75_mlp_pair_v2.py", "nf4_scale_pair.py", "sm75_gluon.py",
                     "benchmark_sm75_mlp_pair_v2.py", "benchmark_mlp_grad_pair.py", "nf4.py"):
        report["sources"][filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    torch.manual_seed(954)
    weights = []
    for _ in range(2):
        dense = torch.randn((8960, 1536), device="cuda", dtype=torch.float16) * .02
        packed, state = bnb.quantize_4bit(dense, blocksize=64, compress_statistics=True, quant_type="nf4")
        packed.quant_state = state
        weights.append(packed)
        del dense
    gate, up = weights
    for m in args.rows:
        dg = torch.randn((m, 8960), device="cuda", dtype=torch.float16)
        du = torch.randn_like(dg)
        rng = torch.cuda.get_rng_state()
        for config in configs:
            case = {"m": m, "n": 1536, "k": 8960, "config": config, "tile": CONFIGS[config]}
            report["cases"].append(case)
            candidate = lambda: sm75_mlp_grad_pair_v2(dg, du, gate, up, config=config)
            reference = lambda: reference_output(dg, du, gate, up, torch.float32)
            scale_only = lambda: predecode_nested_scales_pair(gate.quant_state, up.quant_state)
            try:
                torch.cuda.synchronize()
                began = time.perf_counter()
                actual, compiled = sm75_mlp_grad_pair_v2(dg, du, gate, up, config=config,
                                                        return_compiled=True)
                torch.cuda.synchronize()
                case["compile_plus_first_call_ms"] = (time.perf_counter() - began) * 1000
                ptx = compiled.asm["ptx"]
                case["mma_instruction_count"] = ptx.count("mma.sync.aligned.m16n8k8")
                case["registers"] = getattr(compiled, "n_regs", None)
                case["spills"] = getattr(compiled, "n_spills", None)
                case["shared_bytes"] = getattr(compiled.metadata, "shared", None)
                if case["mma_instruction_count"] == 0:
                    raise RuntimeError("Turing Tensor-Core MMA absent from generated PTX")
                gate_part, up_part = reference_parts(dg, du, gate, up)
                expected = gate_part.float() + up_part.float()
                case["accuracy"] = accuracy_metrics(actual, expected, gate_part, up_part)
                del actual, expected, gate_part, up_part
                if not case["accuracy"]["qualified"] and not args.time_unqualified:
                    case["status"] = "accuracy_failed_not_timed"
                    print(json.dumps(case), flush=True)
                    save()
                    continue
                for _ in range(3):
                    reference()
                    candidate()
                samples = {"reference": [], "candidate": []}
                for _ in range(args.rounds):
                    for name, function in (("reference", reference), ("candidate", candidate),
                                           ("candidate", candidate), ("reference", reference)):
                        samples[name].extend(_sample(function) for _ in range(args.samples))
                case["samples"] = samples
                case["summary"] = {name: _summary(values) for name, values in samples.items()}
                case["scale_predecode_only_summary"] = _summary(
                    [_sample(scale_only) for _ in range(args.samples)])
                case["memory"] = {"reference": _memory(reference), "candidate": _memory(candidate),
                                  "scale_predecode_only": _memory(scale_only)}
                case["rng_unchanged"] = bool(torch.equal(rng, torch.cuda.get_rng_state()))
                case["status"] = ("qualified_microbenchmark_only" if case["accuracy"]["qualified"]
                                  else "accuracy_failed_research_timings_only")
            except Exception as error:
                case.update(status="failure", error=f"{type(error).__name__}: {error}")
                save()
                if type(error).__name__ not in ("OutOfResources", "CompilationError", "UnsupportedLanguageConstruct"):
                    raise
            print(json.dumps({key: value for key, value in case.items() if key != "samples"}), flush=True)
            save()
        del dg, du
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()

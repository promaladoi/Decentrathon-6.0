"""Balanced private Q/V LoRA diagnostic; this is not a full training score.

Both variants draw native dropout RNG on the same full FP32 shape, use the
same GEMMs and keep the same packed masks. The reference has a separate AMP
input cast; the candidate folds it into the existing gather/apply store.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


def _summary(values):
    ordered = sorted(values)
    return {"count": len(values), "median_ms": statistics.median(values),
            "minimum_ms": ordered[0], "maximum_ms": ordered[-1], "values_ms": values}


def _bits(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].view(integer), expected[finite].view(integer))


def _measure(call, samples):
    wall, device = [], []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(samples):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        start.record()
        result = call()
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
        wall.append((time.perf_counter() - begin) * 1000.)
        del result
    return wall, device


def _memory(call):
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return {"initial_allocated_bytes": initial, "peak_allocated_bytes": peak,
            "extra_allocated_peak_bytes": peak - initial}


def run(rounds=3, samples=12, warmups=5):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Triton required")
    if rounds < 2 or samples < 1 or warmups < 1:
        raise ValueError("use at least two rounds, one sample and one warmup")
    from .unpad import _Batch, _lora_out

    torch.manual_seed(897)
    report = {"kind": "private_dropout_linear_cast_diagnostic_not_training_score",
              "training_score_claim": False, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "rounds": rounds, "samples_per_block": samples,
              "warmups": warmups, "ordering": "ABBA each round; A=separate cast, B=fused cast",
              "rng": "reset only in untimed parity; native generator advances during timings",
              "memory": "PyTorch extra allocated peak for one Q/V projection pair, not whole-run NVML",
              "sources_sha256": {}, "cases": []}
    for name in ("dropout.py", "unpad.py", "benchmark_dropout_linear_input.py"):
        report["sources_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for sequence, lengths in ((384, (384, 192, 128, 76)),
                              (512, (512, 250, 100, 22)),
                              (512, (512, 512, 512, 512))):
        batch, width, rank = 4, 1536, 16
        mask = torch.arange(sequence, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]
        rows = mask.reshape(-1).nonzero().squeeze(1)
        x = torch.randn((rows.numel(), width), device="cuda", requires_grad=True)
        aq = torch.nn.Linear(width, rank, bias=False, device="cuda")
        bq = torch.nn.Linear(rank, width, bias=False, device="cuda")
        av = torch.nn.Linear(width, rank, bias=False, device="cuda")
        bv = torch.nn.Linear(rank, 256, bias=False, device="cuda")
        dropout = torch.nn.Dropout(.05).train()
        parameters = (x, aq.weight, bq.weight, av.weight, bv.weight)
        upstream = (torch.randn((rows.numel(), width), device="cuda", dtype=torch.float16) * 16.,
                    torch.randn((rows.numel(), 256), device="cuda", dtype=torch.float16) * 16.)

        def evaluate(candidate, include_outputs=False):
            step = _Batch(rows, batch, sequence)
            with torch.autocast("cuda", dtype=torch.float16):
                if candidate:
                    q = _lora_out((None, aq, bq, dropout, 2.), x, step)
                    v = _lora_out((None, av, bv, dropout, 2.), x, step)
                else:
                    q = bq(aq(step.dropout(dropout, x))) * 2.
                    v = bv(av(step.dropout(dropout, x))) * 2.
            step.done_with(x)
            gradients = torch.autograd.grad((q, v), parameters, upstream)
            return ((q, v), gradients) if include_outputs else gradients

        before = torch.cuda.get_rng_state()
        expected = evaluate(False, True)
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        actual = evaluate(True, True)
        assert torch.equal(torch.cuda.get_rng_state(), after)
        for group_actual, group_expected in zip(actual, expected):
            for value_actual, value_expected in zip(group_actual, group_expected):
                _bits(value_actual, value_expected)
        del expected, actual
        for _ in range(warmups):
            evaluate(False)
            evaluate(True)
        case = {"batch": batch, "sequence": sequence, "width": width, "rank": rank,
                "lengths": list(lengths), "real_rows": rows.numel(), "parity": "exact finite bits, outputs/grads/RNG",
                "separate": {"wall": [], "cuda": []}, "fused": {"wall": [], "cuda": []}, "blocks": []}
        for iteration in range(rounds):
            for candidate in (False, True, True, False):
                wall, device = _measure(lambda: evaluate(candidate), samples)
                name = "fused" if candidate else "separate"
                case[name]["wall"].extend(wall)
                case[name]["cuda"].extend(device)
                case["blocks"].append({"round": iteration, "variant": name,
                                       "wall": _summary(wall), "cuda": _summary(device)})
        for candidate, name in ((False, "separate"), (True, "fused")):
            case[name] = {"wall": _summary(case[name]["wall"]),
                          "cuda": _summary(case[name]["cuda"]),
                          "memory": _memory(lambda: evaluate(candidate))}
        case["cuda_reduction_percent"] = 100. * (1. - case["fused"]["cuda"]["median_ms"] /
                                                 case["separate"]["cuda"]["median_ms"])
        report["cases"].append(case)
        del evaluate, parameters, upstream, x, aq, bq, av, bv, rows, mask, dropout
        gc.collect()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    result = run(rounds=args.rounds, samples=args.samples)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": [
        {"sequence": case["sequence"], "real_rows": case["real_rows"],
         "cuda_reduction_percent": case["cuda_reduction_percent"]} for case in result["cases"]]}, indent=2))

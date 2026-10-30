"""Balanced real-bnb MLP diagnostic; never an official training benchmark.

Copy this file and the two candidate files into solution/, then call
solution.benchmark_mlp_variants.run_tuning() in a dedicated Kaggle T4 process.
The class patches are extracted as callable closures and immediately restored.
The verified NF4 decode patch is shared by all variants and restored on exit.
Inputs and upstream gradients are allocated once, outside every timed region.
"""

import gc
import hashlib
from pathlib import Path
import statistics
import time

import torch


def _summary(values):
    ordered = sorted(values)

    def percentile(p):
        point = (len(ordered) - 1) * p
        low = int(point)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (point - low)

    return {"samples": len(values), "median_ms": statistics.median(values),
            "q25_ms": percentile(0.25), "q75_ms": percentile(0.75),
            "minimum_ms": min(values), "maximum_ms": max(values)}


def _forward_functions():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from . import compressed_swiglu, narrow_mlp
    from .swiglu import swiglu

    def example(mlp, x):
        return mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))

    original = Qwen2MLP.forward
    try:
        compressed_swiglu.patch_qwen2_mlp()
        compressed = Qwen2MLP.forward
        Qwen2MLP.forward = original
        narrow_mlp.patch_qwen2_mlp()
        narrow = Qwen2MLP.forward
    finally:
        Qwen2MLP.forward = original
    return {"example": example, "compressed": compressed, "narrow": narrow}


def _make_mlp(hidden, intermediate):
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    config = Qwen2Config(hidden_size=hidden, intermediate_size=intermediate,
                         num_hidden_layers=1, num_attention_heads=1,
                         num_key_value_heads=1, hidden_act="silu", vocab_size=32)
    mlp = Qwen2MLP(config)
    for name in ("gate_proj", "up_proj", "down_proj"):
        previous = getattr(mlp, name)
        layer = bnb.nn.Linear4bit(previous.in_features, previous.out_features,
                                 bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4")
        # Original random diagnostic weights, not pretrained task weights.
        with torch.no_grad():
            layer.weight.normal_(mean=0.0, std=0.02)
        setattr(mlp, name, layer.to(torch.float16).to("cuda"))
        del previous, layer
    return mlp


def _step(forward, mlp, x, upstream, autocast):
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        output = forward(mlp, x)
    gradient, = torch.autograd.grad(output, (x,), upstream)
    return output, gradient


def _comparison(actual, expected):
    # All compared tensors are fp32; compare storage bits as well as NaN-aware
    # values so overflow diagnostics do not silently mask a payload difference.
    value_equal = torch.all((actual == expected)
                            | (torch.isnan(actual) & torch.isnan(expected)))
    difference = (actual - expected).abs()
    finite = torch.isfinite(difference)
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "reference_dtype": str(expected.dtype),
            "bitwise_equal": bool(torch.equal(actual.view(torch.int32),
                                              expected.view(torch.int32))),
            "value_equal_including_nan": bool(value_equal.item()),
            "max_abs_finite_difference": float(difference[finite].max().item())
            if bool(finite.any().item()) else None,
            "nan_count": int(torch.isnan(actual).sum().item()),
            "inf_count": int(torch.isinf(actual).sum().item())}


def _parity(functions, mlp, x, upstream, autocast):
    from .compressed_swiglu import _eligible

    result = {"eligible": bool(_eligible(mlp, x)), "autocast_fp16": autocast,
              "input_dtype": str(x.dtype), "upstream_dtype": str(upstream.dtype),
              "variants": {}}
    if not result["eligible"]:
        result["status"] = "guard_ineligible_no_candidate_measurement"
        return result
    reference, reference_grad = _step(functions["example"], mlp, x, upstream, autocast)
    result["reference"] = {"output_dtype": str(reference.dtype),
                           "input_gradient_dtype": str(reference_grad.dtype)}
    for name in ("compressed", "narrow"):
        output, gradient = _step(functions[name], mlp, x, upstream, autocast)
        result["variants"][name] = {"output": _comparison(output, reference),
                                      "input_gradient": _comparison(gradient, reference_grad)}
        del output, gradient
    result["all_bitwise_equal"] = all(
        item[key]["bitwise_equal"] for item in result["variants"].values()
        for key in ("output", "input_gradient"))
    result["all_value_equal_including_nan"] = all(
        item[key]["value_equal_including_nan"] for item in result["variants"].values()
        for key in ("output", "input_gradient"))
    result["status"] = "passed" if result["all_bitwise_equal"] else "numerical_difference"
    del reference, reference_grad
    torch.cuda.synchronize()
    return result


def _samples(forward, mlp, x, upstream, autocast, count):
    cuda_values, wall_values = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(count):
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        output, gradient = _step(forward, mlp, x, upstream, autocast)
        end.record()
        end.synchronize()
        wall_values.append(1000.0 * (time.perf_counter() - started))
        cuda_values.append(begin.elapsed_time(end))
        del output, gradient
    return {"cuda_event": _summary(cuda_values), "wall": _summary(wall_values),
            "cuda_event_values_ms": cuda_values, "wall_values_ms": wall_values}


def _memory(forward, mlp, x, upstream, autocast):
    gc.collect()
    torch.cuda.synchronize()
    allocated_start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output, gradient = _step(forward, mlp, x, upstream, autocast)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    allocated_end = torch.cuda.memory_allocated()
    del output, gradient
    torch.cuda.synchronize()
    return {"allocated_start_bytes": allocated_start, "allocated_peak_bytes": peak,
            "extra_allocated_peak_bytes": peak - allocated_start,
            "allocated_end_with_result_bytes": allocated_end,
            "allocated_after_result_release_bytes": torch.cuda.memory_allocated(),
            "meaning": "one MLP forward+input-gradient; allocator allocated bytes, not reserved/NVML"}


def _state_metadata(mlp):
    result = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(mlp, name)
        state = layer.weight.quant_state
        result[name] = {"type": f"{type(layer).__module__}.{type(layer).__qualname__}",
                        "weight_shape": list(state.shape), "weight_dtype": str(state.dtype),
                        "compute_dtype": str(layer.compute_dtype), "nested": state.nested,
                        "quant_type": state.quant_type, "blocksize": state.blocksize,
                        "nested_blocksize": state.state2.blocksize if state.nested else None,
                        "all_parameters_frozen": not any(p.requires_grad for p in layer.parameters())}
    return result


def run_tuning(rounds=3, samples=3, warmups=3, seed=4187):
    """Return serializable balanced AB/BA data; never promote or write a source.

    Defaults keep the diagnostic short: two actual shapes, three paired rounds,
    three samples per block. Compile and allocator warmups precede measurement.
    Wall and CUDA-event spans include the production Python candidate guards.
    """
    if rounds < 3 or samples < 1 or warmups < 1:
        raise ValueError("use >=3 rounds, >=1 sample, >=1 warmup")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 required; no benchmark was run")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"Kaggle T4 required, detected {device_name}")
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import triton
    from .nf4 import patch_nf4_dequant

    started = time.perf_counter()
    result = {"kind": "paired_mlp_diagnostic_not_training_metrics", "device": device_name,
              "torch": torch.__version__, "triton": triton.__version__,
              "bitsandbytes": bnb.__version__, "seed": seed, "rounds": rounds,
              "samples_per_block": samples, "warmups_per_variant": warmups,
              "ordering": "each candidate separately: example,candidate then candidate,example per round",
              "weights": "seeded random diagnostic NF4 weights; same actual bnb layer types and task dimensions",
              "baseline": "v9 organizer SwiGLU plus verified NF4 backward decode",
              "timing": "one full forward+input-gradient, original fp32 input/output; synchronized wall and CUDA events",
              "training_score_claim": False, "quick_parity": [], "cases": [],
              "sources": {}}
    for filename in ("swiglu.py", "nf4.py", "compressed_swiglu.py", "narrow_mlp.py"):
        path = Path(__file__).with_name(filename)
        result["sources"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    original_decode = functional.dequantize_4bit
    try:
        patch_nf4_dequant()
        functions = _forward_functions()
        torch.manual_seed(seed)
        small = _make_mlp(64, 192)
        small_x = torch.randn((2, 31, 64), device="cuda", dtype=torch.float32,
                              requires_grad=True)
        small_grad = torch.randn_like(small_x)
        for autocast in (False, True):
            for scale in (1.0, 256.0, 65536.0):
                upstream = small_grad * scale
                parity = _parity(functions, small, small_x, upstream, autocast)
                parity["upstream_scale"] = scale
                result["quick_parity"].append(parity)
                del upstream
        if not all(item.get("all_value_equal_including_nan", False)
                   for item in result["quick_parity"]):
            result["status"] = "quick_numerical_failure_no_timing"
            return result
        del small, small_x, small_grad
        gc.collect()
        torch.cuda.synchronize()
        mlp = _make_mlp(1536, 8960)
        result["layer_metadata"] = _state_metadata(mlp)
        for sequence in (128, 512):
            x = torch.randn((4, sequence, 1536), device="cuda", dtype=torch.float32,
                            requires_grad=True)
            upstream = torch.randn_like(x) * 256.0
            before_x, before_grad = x.detach().clone(), upstream.clone()
            case = {"shape": [4, sequence, 1536], "intermediate_size": 8960,
                    "autocast_fp16": True, "upstream_scale": 256.0,
                    "parity": _parity(functions, mlp, x, upstream, True),
                    "rounds": [], "memory": {}}
            if not case["parity"].get("all_bitwise_equal", False):
                case["status"] = "numerical_failure_no_timing"
                result["cases"].append(case)
                del x, upstream, before_x, before_grad
                continue
            for _ in range(warmups):
                for name in ("example", "compressed", "narrow"):
                    output, gradient = _step(functions[name], mlp, x, upstream, True)
                    del output, gradient
            torch.cuda.synchronize()
            aggregate = {name: {"cuda_event": [], "wall": []} for name in functions}
            for round_index in range(rounds):
                round_result = {"round": round_index + 1, "pairs": []}
                for candidate in ("compressed", "narrow"):
                    pair = {"candidate": candidate, "blocks": []}
                    for order in (("example", candidate), (candidate, "example")):
                        block = {"order": list(order), "measurements": {}}
                        for name in order:
                            measured = _samples(functions[name], mlp, x, upstream, True, samples)
                            block["measurements"][name] = measured
                            aggregate[name]["cuda_event"].extend(measured["cuda_event_values_ms"])
                            aggregate[name]["wall"].extend(measured["wall_values_ms"])
                        pair["blocks"].append(block)
                    for metric in ("cuda_event", "wall"):
                        pair[f"{metric}_speedup"] = statistics.median([
                            block["measurements"]["example"][metric]["median_ms"]
                            / block["measurements"][candidate][metric]["median_ms"]
                            for block in pair["blocks"]])
                    round_result["pairs"].append(pair)
                case["rounds"].append(round_result)
            case["aggregate"] = {name: {metric: _summary(values)
                                        for metric, values in metrics.items()}
                                  for name, metrics in aggregate.items()}
            for name in ("example", "compressed", "narrow"):
                case["memory"][name] = _memory(functions[name], mlp, x, upstream, True)
            case["inputs_unchanged"] = bool(torch.equal(x, before_x))
            case["upstream_unchanged"] = bool(torch.equal(upstream, before_grad))
            case["status"] = "qualified_diagnostic_only"
            result["cases"].append(case)
            del x, upstream, before_x, before_grad
        result["status"] = ("completed_diagnostic_only" if all(
            case["status"] == "qualified_diagnostic_only" for case in result["cases"])
            else "numerical_failure")
        del mlp
        return result
    finally:
        functional.dequantize_4bit = original_decode
        torch.cuda.synchronize()
        result["diagnostic_wall_seconds"] = time.perf_counter() - started

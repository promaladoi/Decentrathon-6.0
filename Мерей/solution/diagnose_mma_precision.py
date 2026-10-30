"""Explain the inactive Turing paired-GEMM precision failure, without training.

Run from Merey: python solution/diagnose_mma_precision.py --out /tmp/mma_precision.json
The defaults reproduce test_real_mlp_k8960_n1536_keeps_native_partial_ulp_gate:
seed 956, M=33, K=8960, N=1536 and the original default tiles. The strict
qualification function is reused unchanged. This script neither installs a
production patch nor changes any Torch precision flag. Its extra diagnostic
allocations and execution time are not a training memory or speed measurement.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_proof():
    names = ("diagnose_mma_precision.py", "sm75_nf4_gemm.py", "sm75_mlp_pair.py",
             "sm75_gluon.py", "nf4.py", "benchmark_mlp_grad_pair.py",
             "test_mlp_grad_pair.py", "test_sm75_mlp_pair.py")
    return {name: _sha256(ROOT / "solution" / name) for name in names}


def _flags(torch):
    backend = torch.backends.cuda.matmul
    names = ("allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation",
             "allow_bf16_reduced_precision_reduction", "allow_tf32")
    return {name: getattr(backend, name, "unavailable") for name in names}


def _compiled_report(compiled):
    ptx = compiled.asm.get("ptx", "")
    metadata = compiled.metadata
    return {
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "turing_m16n8k8_instructions": len(re.findall(
            r"mma\.sync\.aligned\.m16n8k8\.row\.col\.f32\.f16\.f16\.f32", ptx)),
        "ampere_m16n8k16_instructions": len(re.findall(r"mma\.sync\.aligned\.m16n8k16", ptx)),
        "registers": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_bytes": getattr(metadata, "shared", None),
        "num_warps": getattr(metadata, "num_warps", None),
        "num_stages": getattr(metadata, "num_stages", None),
    }


def _bit_summary(torch, actual, expected):
    integer_type = torch.int16 if actual.dtype == torch.float16 else torch.int32
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    differences = (actual.float() - expected.float()).abs()[finite]
    return {
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "bitwise_equal": bool(torch.equal(actual.view(integer_type), expected.view(integer_type))),
        "different_bits": int((actual.view(integer_type) != expected.view(integer_type)).sum()),
        "different_values": int((actual != expected).sum()),
        "max_finite_abs_error": float(differences.max()) if differences.numel() else 0.0,
    }


def _half_spacing(torch, value):
    magnitude = value.abs()
    return (torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))).float()
            - magnitude.float()).abs()


def _select_samples(torch, pair, independent, native, budget, limit):
    """Prioritize actual gate violations, then fusion differences, then controls."""
    width = native.shape[1]
    error = (pair - native).abs()
    checked = torch.isfinite(error) & torch.isfinite(budget)
    violations = checked & (error > budget)
    fusion_differences = pair.view(torch.int32) != independent.view(torch.int32)
    score = torch.where(checked, error / budget.clamp(min=1e-30),
                        torch.full_like(error, -1.0)).flatten()
    selected = []

    def add_mask(mask, reason):
        indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        if not indices.numel():
            return
        take = min(limit, indices.numel())
        ranked = indices[torch.topk(score[indices], take).indices].cpu().tolist()
        for index in ranked:
            if len(selected) >= limit:
                break
            if all(row[0] != index for row in selected):
                selected.append((index, reason))

    add_mask(violations, "strict_native_partial_ulp_violation")
    add_mask(fusion_differences, "pair_differs_from_sum_of_independent_gluon")
    for row, column in ((0, 0), (native.shape[0] // 2, width // 2),
                        (native.shape[0] - 1, width - 1)):
        index = row * width + column
        if len(selected) < limit and all(item[0] != index for item in selected):
            selected.append((index, "control"))
    return [(index // width, index % width, reason) for index, reason in selected]


def _oracle_samples(torch, coordinates, dg, du, gate_dense, up_dense,
                    native_gate, native_up, independent_gate, independent_up,
                    pair, independent, native, gate_spacing, up_spacing, budget):
    """Only selected decoded-HALF columns and input rows reach CPU FP64.

    A half-by-half product is exactly representable in FP64. math.fsum reduces
    these CPU FP64 products with compensated summation; no full FP64 GPU GEMM
    or alternate quantization decoder is used.
    """
    rows = sorted({row for row, _, _ in coordinates})
    columns = sorted({column for _, column, _ in coordinates})
    row_index = torch.tensor(rows, device=dg.device, dtype=torch.int64)
    column_index = torch.tensor(columns, device=dg.device, dtype=torch.int64)
    dg_rows = dg.index_select(0, row_index).to(device="cpu", dtype=torch.float64).tolist()
    du_rows = du.index_select(0, row_index).to(device="cpu", dtype=torch.float64).tolist()
    gate_columns = gate_dense.index_select(1, column_index).T.to(
        device="cpu", dtype=torch.float64).tolist()
    up_columns = up_dense.index_select(1, column_index).T.to(
        device="cpu", dtype=torch.float64).tolist()
    row_positions = {row: position for position, row in enumerate(rows)}
    column_positions = {column: position for position, column in enumerate(columns)}
    values = {name: value.cpu() for name, value in (
        ("native_gate", native_gate), ("native_up", native_up),
        ("independent_gate", independent_gate), ("independent_up", independent_up),
        ("pair", pair), ("independent_sum", independent), ("native_sum", native),
        ("native_gate_half_spacing", gate_spacing), ("native_up_half_spacing", up_spacing),
        ("strict_native_budget", budget))}
    result = []
    for row, column, reason in coordinates:
        ri, ci = row_positions[row], column_positions[column]
        exact_gate = math.fsum(a * b for a, b in zip(dg_rows[ri], gate_columns[ci]))
        exact_up = math.fsum(a * b for a, b in zip(du_rows[ri], up_columns[ci]))
        rounded = torch.tensor([exact_gate, exact_up], dtype=torch.float64).to(torch.float16).float()
        rounded_sum = float(rounded[0] + rounded[1])
        sample = {"row": row, "column": column, "reason": reason,
                  **{name: float(value[row, column]) for name, value in values.items()},
                  "oracle_gate_float64": exact_gate, "oracle_up_float64": exact_up,
                  "oracle_unrounded_float64_sum": exact_gate + exact_up,
                  "oracle_gate_rounded_half": float(rounded[0]),
                  "oracle_up_rounded_half": float(rounded[1]),
                  "oracle_separately_rounded_half_then_float32_sum": rounded_sum}
        for name in ("native_gate", "native_up", "independent_gate", "independent_up"):
            exact = exact_gate if name.endswith("gate") else exact_up
            half_oracle = float(rounded[0 if name.endswith("gate") else 1])
            sample[name + "_signed_error_vs_unrounded_oracle"] = sample[name] - exact
            sample[name + "_equals_half_oracle"] = sample[name] == half_oracle
        for name in ("pair", "independent_sum", "native_sum"):
            sample[name + "_signed_error_vs_rounded_oracle"] = sample[name] - rounded_sum
        delta = abs(sample["pair"] - sample["native_sum"])
        allowed = sample["strict_native_budget"]
        sample["pair_abs_error_vs_native_sum"] = delta
        sample["pair_error_divided_by_native_budget"] = delta / allowed if allowed else None
        sample["pair_equals_independent_sum"] = sample["pair"] == sample["independent_sum"]
        result.append(sample)
    return {"method": "selected decoded-half rows/columns; CPU FP64 products plus math.fsum",
            "selected_input_rows": rows, "selected_weight_columns": columns,
            "float64_slice_elements": 2 * (len(rows) + len(columns)) * dg.shape[1],
            "samples": result}


def _run(args, report):
    import torch
    import triton

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from solution.benchmark_mlp_grad_pair import accuracy_metrics
    from solution.nf4 import dequantize_nf4
    from solution.sm75_nf4_gemm import TILES, sm75_nf4_gemm
    from solution.sm75_mlp_pair import sm75_mlp_grad_pair
    from solution.test_mlp_grad_pair import _weight

    if not torch.cuda.is_available():
        raise RuntimeError("this standalone diagnostic requires the serialized T4 GPU session")
    device = torch.cuda.current_device()
    flags_before = _flags(torch)
    report.update({"torch": torch.__version__, "triton": triton.__version__,
                   "device": torch.cuda.get_device_name(device),
                   "capability": list(torch.cuda.get_device_capability(device)),
                   "precision_flags_before": flags_before, "source_sha256": _source_proof(),
                   "shape": {"M": args.rows, "K": 8960, "N": 1536}, "seed": args.seed,
                   "general_tile": args.general_tile, "pair_config": args.pair_config,
                   "input_generation": "original test_mlp_grad_pair._weight and test RNG draw order"})
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    try:
        with torch.random.fork_rng(devices=[device]), torch.no_grad():
            # Match manual_seed on CPU/current CUDA device without reseeding other GPUs.
            torch.random.default_generator.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            gate, up = _weight(8960, 1536), _weight(8960, 1536)
            dg = torch.randn((args.rows, 8960), device="cuda", dtype=torch.float16)
            du = torch.randn_like(dg)
            gate_dense = dequantize_nf4(gate, gate.quant_state, torch.float16)
            up_dense = dequantize_nf4(up, up.quant_state, torch.float16)
            native_gate = torch.matmul(dg, gate_dense)
            native_up = torch.matmul(du, up_dense)
            native = native_gate.float() + native_up.float()
            independent_gate, gate_kernel = sm75_nf4_gemm(
                dg, gate, gate.quant_state, transpose_weight=False,
                tile=TILES[args.general_tile], return_compiled=True)
            independent_up, up_kernel = sm75_nf4_gemm(
                du, up, up.quant_state, transpose_weight=False,
                tile=TILES[args.general_tile], return_compiled=True)
            independent = independent_gate.float() + independent_up.float()
            pair, pair_kernel = sm75_mlp_grad_pair(
                dg, du, gate, up, torch.float32, args.pair_config, return_compiled=True)
            torch.cuda.synchronize(device)
            report["strict_original_gate"] = {
                "pair_vs_native": accuracy_metrics(pair, native, native_gate, native_up),
                "sum_independent_vs_native": accuracy_metrics(independent, native, native_gate, native_up),
                "pair_vs_independent_using_independent_partial_budget": accuracy_metrics(
                    pair, independent, independent_gate, independent_up)}
            report["comparisons"] = {
                "pair_vs_sum_independent": _bit_summary(torch, pair, independent),
                "independent_gate_vs_native_gate": _bit_summary(torch, independent_gate, native_gate),
                "independent_up_vs_native_up": _bit_summary(torch, independent_up, native_up)}
            report["compiled"] = {"independent_gate": _compiled_report(gate_kernel),
                                  "independent_up": _compiled_report(up_kernel),
                                  "pair": _compiled_report(pair_kernel)}
            gate_spacing = _half_spacing(torch, native_gate)
            up_spacing = _half_spacing(torch, native_up)
            budget = 2 * (gate_spacing + up_spacing)
            coordinates = _select_samples(torch, pair, independent, native, budget, args.max_samples)
            report["selected_cpu_oracle"] = _oracle_samples(
                torch, coordinates, dg, du, gate_dense, up_dense, native_gate, native_up,
                independent_gate, independent_up, pair, independent, native,
                gate_spacing, up_spacing, budget)
            same = report["comparisons"]["pair_vs_sum_independent"]["bitwise_equal"]
            report["interpretation"] = (
                "Pair equals the separately half-rounded independent Gluon results bitwise. "
                "The native discrepancy is already present in the independent GEMM path; "
                "the ordered pair sum adds no observed difference for these inputs."
                if same else
                "Pair differs from the sum of independent Gluon GEMMs. This isolates a "
                "fusion-specific numerical/code-generation difference for these inputs, "
                "but does not identify its cause or authorize relaxing the strict gate.")
            report["qualification_policy"] = "unchanged; sampled oracle is explanatory, not an admission gate"
    finally:
        report["rng_restored"] = {
            "cpu": bool(torch.equal(cpu_before, torch.random.get_rng_state())),
            "current_cuda": bool(torch.equal(cuda_before, torch.cuda.get_rng_state(device)))}
        report["precision_flags_after"] = _flags(torch)
        report["precision_flags_unchanged"] = flags_before == report["precision_flags_after"]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--seed", type=int, default=956)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--general-tile", default="m32n64k32s1")
    parser.add_argument("--pair-config", default="m32n64k32w4s1")
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.max_samples <= 64:
        parser.error("rows must be positive and max-samples must be between 1 and 64")
    report = {"kind": "mma_precision_diagnostic_not_training_score", "status": "running"}
    exit_code = 0
    try:
        _run(args, report)
        report["status"] = "complete"
    except Exception as error:
        exit_code = 1
        report.update({"status": "error", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output),
                      "pair_equals_independent": report.get("comparisons", {}).get(
                          "pair_vs_sum_independent", {}).get("bitwise_equal"),
                      "rng_restored": report.get("rng_restored")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

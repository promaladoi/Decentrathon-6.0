"""Compare committed original results; never execute or rewrite training code."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess


def compare(repo, ref):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    revision = git("rev-parse", ref).decode().strip()
    paths = git("ls-tree", "-r", "--name-only", "-z", revision).decode().split("\0")
    available = set(paths)
    rows = []
    for path in paths:
        name = PurePosixPath(path)
        if not name.name.endswith("result.json"):
            continue
        check_path = str(name.with_name(name.name.replace("result", "checks")))
        if check_path not in available:
            continue
        raw = git("show", revision + ":" + path)
        result = json.loads(raw)
        checks = json.loads(git("show", revision + ":" + check_path))
        summary = result.get("summary", {})
        solution = summary.get("solution", {})
        base = summary.get("base", {})
        check_rows = checks.get("checks", [])
        digest = hashlib.sha256(raw).hexdigest()
        if not (solution.get("ok") and base.get("ok") and len(check_rows) == 10
                and all(row.get("ok") is True for row in check_rows)
                and checks.get("result_sha256") == digest):
            continue
        ratios = summary["ratios"]
        rows.append({"result_path": path, "checks_path": check_path,
                     "result_sha256": digest, "solution_sha256": checks.get("solution_sha256"),
                     "participant": name.parts[0], "summary": summary,
                     "checks_passed": 10,
                     "reported_loss_nonincreasing": solution["eval_loss"] <= base["eval_loss"],
                     "summary_memory_within_7151_mib": solution["gpu_peak_mib"] <= 7151,
                     "speedup": ratios["speedup"], "memory_ratio": ratios["memory_ratio"]})
    if not rows:
        raise ValueError("No hash-bound ten-check results")
    # Exactly the organizer's >=5% effective-ratio threshold and 35/30/20 weights.
    effective = lambda ratio: ratio if ratio >= 1.05 else 1.0
    best_s = max(effective(row["speedup"]) for row in rows)
    best_m = max(effective(row["memory_ratio"]) for row in rows)
    for row in rows:
        row["illustrative_repository_only_points_out_of_85"] = round(
            20 + (35 * math.log(effective(row["speedup"])) / math.log(best_s) if best_s > 1 else 0)
            + (30 * math.log(effective(row["memory_ratio"])) / math.log(best_m) if best_m > 1 else 0), 2)
    rows.sort(key=lambda row: (-row["illustrative_repository_only_points_out_of_85"], -row["memory_ratio"]))
    protected = {}
    for person in ("Нұржігіт", "Мерей", "Анаис", "Владимир"):
        protected[person] = {folder: git("rev-parse", revision + ":" + person + "/" + folder).decode().strip()
                             for folder in ("harness", "task", "tests")}
    return {"repository": "promaladoi/Decentrathon-6.0", "analyzed_commit": revision,
            "kind": "read-only audit of existing evidence; no new GPU measurement",
            "hash_bound_admitted_report_count_including_duplicates": len(rows),
            "normalizers_from_this_repository_only": {"best_speedup": best_s, "best_memory_ratio": best_m},
            "score_limits": "Illustrative only. Includes 20 correctness points, excludes up to 15 README points. Official normalization uses all admitted teams. Reports include duplicate copies and historical alternatives; report count is not the count of independent teams.",
            "measurement_limits": "Different sessions are not a direct head-to-head performance measurement. Whole-command time differs from training-loop time. Rounded eval loss is not proof on hidden data. Check hashes bind reports, not a fresh execution.",
            "protected_git_trees": protected,
            "protected_trees_equal": all(value == protected["Нұржігіт"] for value in protected.values()),
            "fastest_whole_command_report": min(rows, key=lambda row: row["summary"]["solution"]["wall_s"])["result_path"],
            "lowest_memory_report": min(rows, key=lambda row: row["summary"]["solution"]["gpu_peak_mib"])["result_path"],
            "highest_illustrative_score_report": rows[0]["result_path"],
            "vladimir_evidence_limit": "Current README documents speedup 1.738x, memory ratio 3.469x and 10/10 from pasted output, without current raw result/checks JSON and whole-command seconds. Excluded from hash-bound score ranking; not evidence of a failed solution.",
            "reports": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
    audit = compare(repo, args.ref)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "reports"}, ensure_ascii=True))

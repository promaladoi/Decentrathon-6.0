"""Compare unchanged harness reports, without inventing jury points.

Usage: python 03_validation/compare_variants.py Codex=reports/codex Anais=reports/anais
Each directory must contain the original result.json and checks.json files.
"""

import argparse
import hashlib
import json
from pathlib import Path


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", metavar="NAME=DIRECTORY")
    args = parser.parse_args()
    print("| Version | Checks | Report hash | Eligible checks | Time ratio (base/ours) | Memory ratio | Eval loss delta, % |")
    print("|---|---|---|---|---|---|---|")
    for argument in args.variants:
        if "=" not in argument:
            parser.error("Use NAME=DIRECTORY for every variant")
        name, directory = argument.split("=", 1)
        path = Path(directory)
        raw = (path / "result.json").read_bytes()
        result = json.loads(raw)
        report = json.loads((path / "checks.json").read_text(encoding="utf-8"))
        checks = report.get("checks", [])
        passed = sum(check.get("ok") is True for check in checks)
        matches = report.get("result_sha256") == hashlib.sha256(raw).hexdigest()
        eligible = len(checks) == 10 and passed == 10 and matches
        ratios = result.get("summary", {}).get("ratios", {})
        cells = [name.replace("|", "/"), f"{passed}/{len(checks)}",
                 "MATCH" if matches else "MISMATCH", "YES" if eligible else "NO",
                 number(ratios.get("speedup")), number(ratios.get("memory_ratio")),
                 number(ratios.get("quality_delta_pct"))]
        print("| " + " | ".join(cells) + " |")
    print("\nEligibility here means the supplied 10 checks passed and match the report bytes. "
          "It does not replace source review or the hidden-data jury run. "
          "Different reports must use the same fixed task and Kaggle T4. "
          "No official points are computed: those depend on all admitted teams.")


if __name__ == "__main__":
    main()

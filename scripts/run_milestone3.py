"""Run from the repository root after installing the dense dependencies."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="New experiment name, e.g. student-m3-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        parser.error("Use letters, numbers, underscores and hyphens for the experiment name")
    baseline = Path("runs") / f"{args.name}-bm25"
    candidate = Path("runs") / f"{args.name}-dense"
    comparison = Path("comparisons") / f"{args.name}.json"
    if any(path.exists() for path in [baseline, candidate, comparison]):
        parser.error("That experiment name already has outputs; choose a new name")
    commands = [
        ["verify-split"],
        ["model-download"],
        ["build-dense"],
        ["evaluate", "--system", "bm25", "--output", str(baseline)],
        ["evaluate", "--system", "dense", "--output", str(candidate)],
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(comparison),
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", "retrieval_lab.cli", *command], check=True)


if __name__ == "__main__":
    main()

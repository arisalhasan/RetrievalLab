"""Sequential same-split BM25/dense/hybrid experiments. Run from the repository root."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from retrieval_lab.hybrid import HybridConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="New experiment prefix, e.g. student-m4")
    parser.add_argument("--hybrid-config", type=Path, default=Path("configs/hybrid.json"))
    parser.add_argument("--repeat", action="store_true", help="Repeat all three in fresh processes")
    parser.add_argument(
        "--depth-ablation",
        type=int,
        help="Also run a declared alternative candidate depth; no automatic selection",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        parser.error("Use letters, numbers, underscores and hyphens for the experiment name")
    config = HybridConfig.from_file(args.hybrid_config)
    if config.candidate_depth < 10:
        parser.error("Validation requires at least 10 candidates per source")
    if args.depth_ablation is not None and (
        args.depth_ablation < 10 or args.depth_ablation == config.candidate_depth
    ):
        parser.error("The ablation depth must be >= 10 and differ from the main candidate depth")
    settings = {"bm25": config.bm25, "dense": config.dense, "hybrid": config}
    if args.depth_ablation is not None:
        settings[f"hybrid-depth{args.depth_ablation}"] = replace(
            config, candidate_depth=args.depth_ablation
        )
    config_dir = Path("runs") / f"{args.name}-configs"
    stages = ["v1", "reproduction"] if args.repeat else ["v1"]
    runs = {
        (stage, system): Path("runs") / f"{args.name}-{system}-validation-{stage}"
        for stage in stages
        for system in settings
    }
    outputs = {
        (stage, baseline): Path("comparisons") / f"{args.name}-{baseline}-vs-hybrid-{stage}.json"
        for stage in stages
        for baseline in ["bm25", "dense"]
    }
    if args.depth_ablation is not None:
        ablation = f"hybrid-depth{args.depth_ablation}"
        for stage in stages:
            outputs[stage, ablation] = (
                Path("comparisons") / f"{args.name}-{ablation}-vs-hybrid-{stage}.json"
            )
    if any(path.exists() for path in [config_dir, *runs.values(), *outputs.values()]):
        parser.error(
            "That name already has outputs; choose a new name to preserve earlier evidence"
        )
    config_dir.mkdir(parents=True)
    # Snapshot the exact same children for standalone baselines and the fused system.
    for system, values in settings.items():
        (config_dir / f"{system}.json").write_text(
            json.dumps(values.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def run(*command):
        subprocess.run([sys.executable, "-m", "retrieval_lab.cli", *map(str, command)], check=True)

    run("verify-split")
    run("model-download", "--config", config_dir / "dense.json")
    run("build-dense", "--config", config_dir / "dense.json")
    for stage in stages:
        for system, values in settings.items():
            run(
                "evaluate",
                "--system",
                values.system,
                "--retriever-config",
                config_dir / f"{system}.json",
                "--output",
                runs[stage, system],
            )
        for baseline in [name for step, name in outputs if step == stage]:
            run(
                "compare",
                "--baseline",
                runs[stage, baseline],
                "--candidate",
                runs[stage, "hybrid"],
                "--output",
                outputs[stage, baseline],
            )


if __name__ == "__main__":
    main()

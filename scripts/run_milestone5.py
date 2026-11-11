"""Same-split reranking experiments: full quality and a declared timing sample."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from retrieval_lab.evaluation import EvaluationConfig
from retrieval_lab.reranking import RerankerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="New experiment prefix, e.g. student-m5")
    parser.add_argument("--reranker-config", type=Path, default=Path("configs/reranker.json"))
    parser.add_argument(
        "--evaluation-config", type=Path, default=Path("configs/evaluation-m5.json")
    )
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        parser.error("Use letters, numbers, underscores and hyphens for the experiment name")
    config = RerankerConfig.from_file(args.reranker_config)
    evaluation = EvaluationConfig.from_file(args.evaluation_config)
    if config.rerank_depth < evaluation.cutoff:
        parser.error("rerank_depth must be at least the final cutoff")
    settings = {
        "bm25": config.retriever.bm25,
        "dense": config.retriever.dense,
        "hybrid": config.retriever,
        "rerank": config,
    }
    stages = ["v1", "reproduction"] if args.repeat else ["v1"]
    config_dir = Path("runs") / f"{args.name}-configs"
    runs = {
        (stage, name): Path("runs") / f"{args.name}-{name}-validation-{stage}"
        for stage in stages
        for name in settings
    }
    comparisons = {
        (stage, name): Path("comparisons") / f"{args.name}-{name}-vs-rerank-{stage}.json"
        for stage in stages
        for name in ["bm25", "dense", "hybrid"]
    }
    if any(p.exists() for p in [config_dir, *runs.values(), *comparisons.values()]):
        parser.error("That experiment name already has outputs; choose a new name")
    config_dir.mkdir(parents=True)
    for name, values in settings.items():
        (config_dir / f"{name}.json").write_text(
            json.dumps(values.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    evaluation_path = config_dir / "evaluation.json"
    evaluation_path.write_bytes(args.evaluation_config.read_bytes())

    def run(*command):
        subprocess.run([sys.executable, "-m", "retrieval_lab.cli", *map(str, command)], check=True)

    run("verify-split")
    run("model-download", "--config", config_dir / "dense.json")
    run("reranker-download", "--config", config_dir / "rerank.json")
    run("build-dense", "--config", config_dir / "dense.json")
    for stage in stages:
        for name in settings:
            run(
                "evaluate",
                "--system",
                name,
                "--retriever-config",
                config_dir / f"{name}.json",
                "--evaluation-config",
                evaluation_path,
                "--output",
                runs[stage, name],
            )
        for name in ["bm25", "dense", "hybrid"]:
            run(
                "compare",
                "--baseline",
                runs[stage, name],
                "--candidate",
                runs[stage, "rerank"],
                "--output",
                comparisons[stage, name],
            )


if __name__ == "__main__":
    main()

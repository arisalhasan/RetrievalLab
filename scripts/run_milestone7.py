"""New train/validation experiment; preserves the original release and never reads test labels."""

import argparse
import json
from pathlib import Path

from retrieval_lab.evaluation import run_evaluation
from retrieval_lab.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Experiment directory exists; use a new name")
    training_output = args.output / "training"
    result = train(
        Path("data/scifact"),
        Path("configs/splits/scifact-train-v1.json"),
        Path("configs/reranker.json"),
        Path("configs/training.json"),
        training_output,
    )
    run_evaluation(
        Path("data/scifact"),
        Path("configs/splits/scifact-train-v1.json"),
        training_output / "best-trained-config.json",
        Path("configs/evaluation-m5.json"),
        args.output / "validation",
        system="rerank",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

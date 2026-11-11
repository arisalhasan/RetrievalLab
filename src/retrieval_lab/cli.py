"""Small commands for data preparation and a recorded BM25 smoke experiment."""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from retrieval_lab import __version__
from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.data import DatasetError, load_dataset
from retrieval_lab.download import download_scifact
from retrieval_lab.provenance import code_state, source_hashes
from retrieval_lab.splits import load_split, make_split, write_split


def run_search(data_dir: Path, config_path: Path, query: str, top_k: int) -> dict:
    """Run on the train split. This is an interactive search record, not evaluation."""
    if not query.strip():
        raise ValueError("Query must contain some text")
    config = BM25Config.from_file(config_path)
    dataset = load_dataset(data_dir, split="train")
    started = perf_counter()
    index = BM25Index(list(dataset.corpus.values()), config)
    build_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    results = index.search(query, top_k)
    search_ms = (perf_counter() - started) * 1000
    return {
        "schema_version": 1,
        "run_type": "single_query_smoke_test_not_benchmark",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code": code_state(),
        "source_file_sha256": source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "retrievallab": __version__,
            "streamlit": importlib.metadata.version("streamlit"),
        },
        "dataset": {
            "name": "BEIR/SciFact",
            "split": dataset.split,
            "file_sha256": dataset.file_sha256,
            "documents": len(dataset.corpus),
        },
        "config": config.to_dict(),
        "query": query,
        "top_k": top_k,
        "timing_ms": {"index_build": build_ms, "query_only": search_ms},
        "timing_scope": "one cold query; excludes data loading and UI; no percentile estimate",
        "results": [
            {
                "rank": rank,
                "doc_id": hit.document.doc_id,
                "title": hit.document.title,
                "score": hit.score,
            }
            for rank, hit in enumerate(results, 1)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RetrievalLab search and validation experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify official BEIR SciFact")
    download.add_argument("--data-root", type=Path, default=Path("data"))
    download.add_argument("--archive", type=Path, help="Use an already-downloaded official ZIP")
    search = commands.add_parser("search", help="Search SciFact and optionally save a run record")
    search.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    search.add_argument("--config", type=Path, default=Path("configs/bm25.json"))
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--output", type=Path)
    split = commands.add_parser("split", help="Save a grouped split of official training queries")
    split.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    verify = commands.add_parser("verify-split", help="Verify saved IDs, grouping and input hashes")
    verify.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    verify.add_argument(
        "--manifest", type=Path, default=Path("configs/splits/scifact-train-v1.json")
    )
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a frozen retriever on train-derived validation"
    )
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    evaluate.add_argument(
        "--split-manifest", type=Path, default=Path("configs/splits/scifact-train-v1.json")
    )
    evaluate.add_argument("--system", choices=["bm25", "dense", "hybrid", "rerank"], default="bm25")
    evaluate.add_argument("--retriever-config", "--bm25-config", type=Path)
    evaluate.add_argument("--model-root", type=Path, default=Path("models"))
    evaluate.add_argument("--cache-root", type=Path, default=Path("indexes"))
    evaluate.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.json"))
    evaluate.add_argument("--output", type=Path, required=True)
    model = commands.add_parser(
        "model-download", help="Download and verify the pinned public encoder"
    )
    model.add_argument("--config", type=Path, default=Path("configs/dense.json"))
    model.add_argument("--model-root", type=Path, default=Path("models"))
    reranker = commands.add_parser(
        "reranker-download", help="Download the pinned pretrained TinyBERT reranker"
    )
    reranker.add_argument("--config", type=Path, default=Path("configs/reranker.json"))
    reranker.add_argument("--model-root", type=Path, default=Path("models"))
    dense = commands.add_parser("build-dense", help="Prepare a verified corpus embedding cache")
    dense.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    dense.add_argument("--config", type=Path, default=Path("configs/dense.json"))
    dense.add_argument("--model-root", type=Path, default=Path("models"))
    dense.add_argument("--cache-root", type=Path, default=Path("indexes"))
    compare = commands.add_parser(
        "compare", help="Compare verified runs on the same validation IDs"
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(f"Verified SciFact: {download_scifact(args.data_root, args.archive).resolve()}")
        elif args.command == "split":
            manifest = make_split(
                load_dataset(args.data_dir, "train"),
                args.validation_fraction,
                args.seed,
                args.near_duplicate_threshold,
            )
            write_split(manifest, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "counts": manifest["counts"],
                        "split_sha256": manifest["split_sha256"],
                    },
                    indent=2,
                )
            )
        elif args.command == "verify-split":
            manifest = load_split(args.manifest, load_dataset(args.data_dir, "train"))
            print(
                json.dumps(
                    {
                        "verified": True,
                        "counts": manifest["counts"],
                        "split_sha256": manifest["split_sha256"],
                    },
                    indent=2,
                )
            )
        elif args.command == "model-download":
            from retrieval_lab.dense_config import DenseConfig
            from retrieval_lab.model_store import download_model

            path = download_model(DenseConfig.from_file(args.config), args.model_root)
            print(f"Verified local model: {path}")
        elif args.command == "reranker-download":
            from retrieval_lab.reranker_store import download_model
            from retrieval_lab.reranking import RerankerConfig

            path = download_model(RerankerConfig.from_file(args.config), args.model_root)
            print(f"Verified local reranker: {path}")
        elif args.command == "build-dense":
            from retrieval_lab.dense import build_dense
            from retrieval_lab.dense_config import DenseConfig
            from retrieval_lab.profiling import ResourceMonitor

            dataset = load_dataset(args.data_dir, "train")
            with ResourceMonitor() as preparation:
                index = build_dense(
                    list(dataset.corpus.values()),
                    DenseConfig.from_file(args.config),
                    args.model_root,
                    args.cache_root,
                )
            print(
                json.dumps(
                    {"preparation": index.preparation, "resources": preparation.result}, indent=2
                )
            )
        elif args.command == "compare":
            from retrieval_lab.comparison import compare_runs

            result = compare_runs(args.baseline, args.candidate, args.data_dir, args.output)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "per_query"}, indent=2
                )
            )
        elif args.command == "evaluate":
            from retrieval_lab.evaluation import run_evaluation

            report = run_evaluation(
                args.data_dir,
                args.split_manifest,
                args.retriever_config or Path(f"configs/{args.system}.json"),
                args.evaluation_config,
                args.output,
                args.system,
                args.model_root,
                args.cache_root,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "quality": report["quality"],
                        "latency": report["latency"],
                        "reference_check": report["reference_check"],
                    },
                    indent=2,
                )
            )
        else:
            record = run_search(args.data_dir, args.config, args.query, args.top_k)
            rendered = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects earlier experiment evidence from replacement.
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(rendered)
            print(rendered, end="")
        return 0
    except (DatasetError, OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

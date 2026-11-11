"""Seal development choices before an explicit, auditable official test evaluation."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from retrieval_lab.data import OFFICIAL_FILES, DatasetError, load_dataset, sha256_file
from retrieval_lab.evaluation import EvaluationConfig, evaluate, load_report, save_evaluation
from retrieval_lab.provenance import environment, source_hashes
from retrieval_lab.retrievers import load_config
from retrieval_lab.splits import canonical_hash


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DatasetError("Release inputs must remain inside the project directory")
    return path


def freeze(spec_path: Path, output: Path, root: Path = Path(".")) -> dict:
    """Read TRAIN labels/reports only. Hashing test bytes does not parse any test judgment."""
    if output.exists():
        raise FileExistsError("Release freeze already exists; it will not be overwritten")
    spec = _read(spec_path)
    dataset = load_dataset(root / "data/scifact", "train")
    from retrieval_lab.dense_config import DenseConfig
    from retrieval_lab.model_store import model_folder as dense_folder
    from retrieval_lab.model_store import verify_model as verify_dense
    from retrieval_lab.reranker_store import model_folder, verify_model
    from retrieval_lab.reranking import RerankerConfig

    records, hashes = {}, {}
    split_id = None
    for name, entry in spec["systems"].items():
        if not name.replace("-", "").isalnum():
            raise ValueError("Use alphanumeric/hyphen system names")
        config_path = _inside(root, entry["config"])
        config = load_config(config_path, entry["system"])
        folder = _inside(root, entry["validation_run"])
        summary, _, _ = load_report(folder, dataset)
        if summary["config"]["retriever"] != config.to_dict():
            raise DatasetError("Validation configuration differs from the release candidate")
        if split_id is not None and split_id != summary["split_sha256"]:
            raise DatasetError("Release candidates must use the same development split")
        split_id = summary["split_sha256"]
        for path in [
            config_path,
            *(
                folder / n
                for n in ["manifest.json", "split.json", "queries.json", "timings.csv", "report.md"]
            ),
        ]:
            hashes[path.resolve().relative_to(root.resolve()).as_posix()] = sha256_file(path)
        models = []
        if isinstance(config, RerankerConfig):
            models.append(
                (model_folder(config, root / "models"), verify_model(config, root / "models"))
            )
            dense = config.retriever.dense
        else:
            dense = getattr(config, "dense", config)
        if isinstance(dense, DenseConfig):
            models.append(
                (dense_folder(dense, root / "models"), verify_dense(dense, root / "models"))
            )
        for model_dir, manifest in models:
            for filename in [*manifest["file_sha256"], "retrievallab-model.json"]:
                path = model_dir / filename
                hashes[path.resolve().relative_to(root.resolve()).as_posix()] = sha256_file(path)
        records[name] = {
            **entry,
            "retriever_config": config.to_dict(),
            "validation_quality": summary["quality"],
        }
    if not records:
        raise ValueError("At least one verified release candidate is required")
    # This rule is declared by code before official test-label access. Never promote by test score.
    selected = max(
        sorted(records),
        key=lambda n: (
            records[n]["validation_quality"]["ndcg"],
            records[n]["validation_quality"]["recall"],
        ),
    )
    evaluation_path = _inside(root, spec["evaluation_config"])
    evaluation = EvaluationConfig.from_file(evaluation_path)
    hashes[spec["evaluation_config"]] = sha256_file(evaluation_path)
    for name in spec.get("extra_inputs", []):
        hashes[name] = sha256_file(_inside(root, name))
    payload = {
        "schema_version": 1,
        "stage": "pre_test_release_freeze",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "systems": records,
        "selected_system": selected,
        "selection_rule": "Highest full-validation nDCG@10, then Recall@10, "
        "then lexicographically first name; exploratory validation, no test-based promotion",
        "evaluation_config": evaluation.__dict__,
        "development_split_sha256": split_id,
        "test_protocol": "All official test queries; fixed seed-based timing sample "
        "and fresh inference. "
        "Query bootstrap (1000 resamples); dependent claims may understate uncertainty. "
        "Fresh per-system Python workers via the final CLI. Descriptive single-workstation "
        "latency/resources. No subsequent tuning on test.",
        "dataset_file_sha256": {n: sha256_file(root / "data/scifact" / n) for n in OFFICIAL_FILES},
        "input_file_sha256": hashes,
        "source_file_sha256": source_hashes(),
        "environment": environment(),
        "test_labels_parsed_at_freeze": False,
    }
    result = {**payload, "freeze_sha256": canonical_hash(payload)}
    _write_new(output, result)
    return result


def verify_freeze(path: Path, root: Path = Path(".")) -> dict:
    manifest = _read(path)
    payload = {k: v for k, v in manifest.items() if k != "freeze_sha256"}
    if canonical_hash(payload) != manifest["freeze_sha256"]:
        raise DatasetError("Release freeze checksum mismatch")
    if manifest["source_file_sha256"] != source_hashes():
        raise DatasetError("Source changed after freeze; official evaluation refused")
    if manifest["environment"] != environment():
        raise DatasetError("Runtime changed after freeze; official evaluation refused")
    for name, expected in manifest["input_file_sha256"].items():
        if sha256_file(_inside(root, name)) != expected:
            raise DatasetError(f"Frozen release input changed: {name}")
    for name, expected in manifest["dataset_file_sha256"].items():
        if sha256_file(root / "data/scifact" / name) != expected:
            raise DatasetError("Official dataset changed after freeze")
    return manifest


def test_partition(dataset, freeze_sha256: str) -> dict:
    if dataset.split != "test":
        raise DatasetError("Expected the explicit official test dataset")
    ids = sorted(dataset.queries)
    payload = {
        "schema_version": 1,
        "source_split": "test",
        "freeze_sha256": freeze_sha256,
        "source_file_sha256": dataset.file_sha256,
        # Legacy key permits identical metric/timing code without disguising dataset.split.
        "validation_query_ids": ids,
        "train_query_ids": [],
        "groups": {qid: [qid] for qid in ids},
        "counts": {"test": len(ids), "groups": len(ids)},
    }
    return {**payload, "split_sha256": canonical_hash(payload)}


def run_final(
    freeze_path: Path,
    output: Path,
    root: Path = Path("."),
    *,
    isolated_processes: bool = False,
    only: str | None = None,
) -> dict:
    manifest = verify_freeze(freeze_path, root)
    output.mkdir(parents=True, exist_ok=True)
    # Recorded BEFORE the first parse, including failed attempts. Completed runs never overwritten.
    with (output / "test-access.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "explicit_official_test_label_access",
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "freeze_sha256": manifest["freeze_sha256"],
                }
            )
            + "\n"
        )
    dataset = load_dataset(root / "data/scifact", "test")
    split = test_partition(dataset, manifest["freeze_sha256"])
    results = {}
    if only is not None and only not in manifest["systems"]:
        raise ValueError("System is not in the frozen release")
    for name, entry in manifest["systems"].items():
        if only is not None and name != only:
            continue
        folder = output / name
        if isolated_processes and not folder.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "retrieval_lab.release",
                    "evaluate-one",
                    "--freeze",
                    str(freeze_path.resolve()),
                    "--output",
                    str(output.resolve()),
                    "--system",
                    name,
                ],
                cwd=root.resolve(),
                check=True,
            )
        if folder.exists():
            summary, _, _ = load_report(folder, dataset)
            if summary.get("freeze_sha256") != manifest["freeze_sha256"]:
                raise DatasetError("Existing final run belongs to another freeze")
        else:
            print(json.dumps({"event": "official_test_started", "system": name}), flush=True)
            summary, queries, timings = evaluate(
                dataset,
                split,
                load_config(_inside(root, entry["config"]), entry["system"]),
                EvaluationConfig(**manifest["evaluation_config"]),
                root / "models",
                root / "indexes",
                official_freeze_sha256=manifest["freeze_sha256"],
            )
            summary["release_system"] = name
            summary["system"] = name
            summary["measurement_protocol"]["process_isolation"] = (
                "fresh per-system Python worker" if only is not None else "in-process library call"
            )
            summary = save_evaluation(summary, queries, timings, split, folder)
            load_report(folder, dataset)
        results[name] = summary["quality"]
        print(
            json.dumps(
                {"event": "official_test_complete", "system": name, "quality": summary["quality"]}
            ),
            flush=True,
        )
    result = {
        "freeze_sha256": manifest["freeze_sha256"],
        "selected_before_test": manifest["selected_system"],
        "quality": results,
        "selection_changed_after_test": False,
    }
    if only is None and not (output / "result.json").exists():
        _write_new(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("freeze")
    seal.add_argument("--spec", type=Path, default=Path("configs/release.json"))
    seal.add_argument("--output", type=Path, default=Path("releases/v1/freeze.json"))
    run = commands.add_parser("evaluate-test")
    run.add_argument("--freeze", type=Path, default=Path("releases/v1/freeze.json"))
    run.add_argument("--output", type=Path, default=Path("runs/m8-official-test-v1"))
    worker = commands.add_parser(
        "evaluate-one", help="Isolated worker; requires an existing verified freeze"
    )
    worker.add_argument("--freeze", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--system", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.spec, args.output)
    else:
        result = run_final(
            args.freeze,
            args.output,
            isolated_processes=args.command == "evaluate-test",
            only=args.system if args.command == "evaluate-one" else None,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

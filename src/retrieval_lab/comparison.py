"""Paired comparisons require identical labels, query IDs and evaluation settings."""

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from retrieval_lab.data import DatasetError, load_dataset, sha256_file
from retrieval_lab.evaluation import cluster_intervals, load_report


def compare_loaded(
    baseline: dict, baseline_rows: list, candidate: dict, candidate_rows: list, split: dict
) -> dict:
    for key in ["dataset", "split_sha256", "metric_definition"]:
        if baseline[key] != candidate[key]:
            raise DatasetError(f"Cannot compare runs with different {key}")
    if baseline["split_sha256"] != split["split_sha256"]:
        raise DatasetError("Comparison split does not match the run identity")
    if baseline["config"]["evaluation"] != candidate["config"]["evaluation"]:
        raise DatasetError("Comparison requires the same evaluation configuration")
    if baseline["measurement_protocol"].get("latency_query_ids") != candidate[
        "measurement_protocol"
    ].get("latency_query_ids"):
        raise DatasetError("Comparison requires the same timing query sample")
    if not all(report["reference_check"]["passed"] for report in [baseline, candidate]):
        raise DatasetError("Both runs must pass reference metric checks")
    left = {row["query_id"]: row for row in baseline_rows}
    right = {row["query_id"]: row for row in candidate_rows}
    if sorted(left) != split["validation_query_ids"] or set(left) != set(right):
        raise DatasetError("Comparison query IDs must equal the frozen validation partition")
    deltas = {
        qid: {metric: right[qid][metric] - left[qid][metric] for metric in ["recall", "ndcg"]}
        for qid in sorted(left)
    }
    config = baseline["config"]["evaluation"]
    intervals = cluster_intervals(
        deltas, split["groups"], config["seed"], config["bootstrap_samples"]
    )
    systems = []
    for role, report in [("baseline", baseline), ("candidate", candidate)]:
        peak = report["resources"]["measured_queries"]["sampled_peak_rss_bytes"] / 2**20
        systems.append(
            {
                "role": role,
                "system": report["system"],
                "recall_at_10": report["quality"]["recall"],
                "ndcg_at_10": report["quality"]["ndcg"],
                "p50_ms": report["latency"]["p50_ms"],
                "p95_ms": report["latency"]["p95_ms"],
                "query_stage_sampled_peak_rss_mib": peak,
                "preparation_seconds": report["resources"]["index_build"]["wall_seconds"],
            }
        )
    outcomes = {
        metric: {
            "candidate_higher": sum(row[metric] > 1e-12 for row in deltas.values()),
            "candidate_lower": sum(row[metric] < -1e-12 for row in deltas.values()),
            "equal": sum(abs(row[metric]) <= 1e-12 for row in deltas.values()),
        }
        for metric in ["recall", "ndcg"]
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "comparison_type": "paired_official_test"
        if baseline["run_type"] == "official_test_evaluation"
        else "paired_train_derived_validation",
        "split_sha256": split["split_sha256"],
        "query_count": len(left),
        "systems": systems,
        "candidate_minus_baseline": {
            metric: fmean(row[metric] for row in deltas.values()) for metric in ["recall", "ndcg"]
        },
        "paired_95pct_intervals": intervals,
        "per_query_outcomes": outcomes,
        "coverage": {
            "candidate_only_found_evidence": sum(
                left[q]["recall"] == 0 and right[q]["recall"] > 0 for q in left
            ),
            "baseline_only_found_evidence": sum(
                right[q]["recall"] == 0 and left[q]["recall"] > 0 for q in left
            ),
            "both_missed_all_evidence": sum(
                left[q]["recall"] == 0 and right[q]["recall"] == 0 for q in left
            ),
        },
        "same_environment_snapshot": baseline["environment"] == candidate["environment"],
        "timing_caveat": "Separate process runs on an uncontrolled workstation. "
        "Timings are descriptive, not a causal or stable speed comparison. "
        "Dense/hybrid/rerank query time includes encoding; rerank includes fresh pair scoring. "
        "Hybrid runs both retrievers and fusion "
        "serially. One-time preparation is separate.",
        "per_query": [
            {
                "query_id": qid,
                "query": left[qid]["query"],
                "baseline_recall": left[qid]["recall"],
                "candidate_recall": right[qid]["recall"],
                "baseline_ndcg": left[qid]["ndcg"],
                "candidate_ndcg": right[qid]["ndcg"],
                "recall_delta": deltas[qid]["recall"],
                "ndcg_delta": deltas[qid]["ndcg"],
            }
            for qid in sorted(left)
        ],
    }


def compare_runs(
    baseline_dir: Path, candidate_dir: Path, data_dir: Path, output: Path | None = None
) -> dict:
    if output is not None and output.exists():
        raise FileExistsError(f"Comparison already exists: {output}")
    dataset = load_dataset(data_dir, "train")
    baseline, left = load_report(baseline_dir, dataset)[:2]
    candidate, right = load_report(candidate_dir, dataset)[:2]
    split = json.loads((baseline_dir / "split.json").read_text(encoding="utf-8"))
    result = compare_loaded(baseline, left, candidate, right, split)
    result["source_runs"] = {
        role: {"name": folder.name, "manifest_sha256": sha256_file(folder / "manifest.json")}
        for role, folder in [("baseline", baseline_dir), ("candidate", candidate_dir)]
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    return result

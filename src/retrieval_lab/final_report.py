"""Export complete held-out evidence to shareable CSV, figures, and a readable report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from retrieval_lab.comparison import compare_loaded
from retrieval_lab.data import load_dataset, sha256_file
from retrieval_lab.evaluation import load_report
from retrieval_lab.exports import comparison_figure
from retrieval_lab.release import verify_freeze


def export_release(freeze_path: Path, runs: Path, output: Path, root: Path = Path(".")):
    if output.exists():
        raise FileExistsError("Report export exists; choose a new directory")
    freeze = verify_freeze(freeze_path, root)
    if not (runs / "result.json").exists():
        raise ValueError("All frozen official test runs must complete before report export")
    dataset = load_dataset(root / "data/scifact", "test")
    reports, rows = {}, []
    for name in freeze["systems"]:
        report, queries, _ = load_report(runs / name, dataset)
        if report.get("freeze_sha256") != freeze["freeze_sha256"]:
            raise ValueError("Cannot mix release freezes")
        reports[name] = (report, queries)
        rows.append(
            {
                "role": "selected before test"
                if name == freeze["selected_system"]
                else "comparison",
                "system": name,
                "recall_at_10": report["quality"]["recall"],
                "ndcg_at_10": report["quality"]["ndcg"],
                "p50_ms": report["latency"]["p50_ms"],
                "p95_ms": report["latency"]["p95_ms"],
                "query_stage_sampled_peak_rss_mib": report["resources"]["measured_queries"][
                    "sampled_peak_rss_bytes"
                ]
                / 2**20,
                "preparation_seconds": report["resources"]["index_build"]["wall_seconds"],
                "process_cpu_seconds": report["resources"]["measured_queries"][
                    "process_cpu_seconds"
                ],
                "missed_queries": report["quality"]["queries_with_no_relevant_retrieved"],
            }
        )
    baseline_name = "bm25" if "bm25" in reports else next(iter(reports))
    baseline, baseline_queries = reports[baseline_name]
    split = json.loads((runs / baseline_name / "split.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "scores.csv", index=False)
    figure = {"systems": rows, "query_count": len(dataset.queries)}
    (output / "comparison.png").write_bytes(comparison_figure(figure))
    (output / "comparison.svg").write_bytes(comparison_figure(figure, "svg"))
    comparisons = {}
    for name, (report, queries) in reports.items():
        if name == baseline_name:
            continue
        compared = compare_loaded(baseline, baseline_queries, report, queries, split)
        compared["paired_95pct_intervals"]["method"] = (
            "paired query bootstrap; singleton test queries; "
            "claim dependence may understate uncertainty"
        )
        comparisons[name] = compared
        (output / f"{baseline_name}-vs-{name}.json").write_text(
            json.dumps(compared, indent=2) + "\n", encoding="utf-8"
        )
    selected = freeze["selected_system"]
    chosen_report, chosen_queries = reports[selected]
    text = [
        "# RetrievalLab — official SciFact test, release v1",
        "",
        f"Measured on {chosen_report['created_at_utc']}. "
        f"All {len(dataset.queries)} official test queries were evaluated.",
        "",
        f"**Selected before test: {selected}.** "
        "The selection remains unchanged after test results.",
        f"Freeze SHA-256: `{freeze['freeze_sha256']}`.",
        "",
        "| System | Recall@10 | nDCG@10 | p50 / p95 ms | Query peak RSS MiB | Complete misses |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        text.append(
            f"| {row['system']} | {row['recall_at_10']:.6f} | {row['ndcg_at_10']:.6f} | "
            f"{row['p50_ms']:.2f} / {row['p95_ms']:.2f} | "
            f"{row['query_stage_sampled_peak_rss_mib']:.2f} | {row['missed_queries']} |"
        )
    text += [
        "",
        "## Uncertainty and paired comparisons",
        "",
        "95% query-bootstrap intervals use 1,000 seed-42 samples. Related claims can make these "
        "intervals too narrow. Differences are exploratory, "
        "without multiple-comparison adjustment. "
        "A point-score advantage is not automatically a reliable improvement.",
        "",
    ]
    for name, (report, _) in reports.items():
        ci = report["confidence_intervals"]
        text.append(
            f"- {name}: Recall [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}], "
            f"nDCG [{ci['ndcg'][0]:.4f}, {ci['ndcg'][1]:.4f}]."
        )
    for name, compared in comparisons.items():
        delta, ci = compared["candidate_minus_baseline"], compared["paired_95pct_intervals"]
        text.append(
            f"- {name} minus {baseline_name}: nDCG {delta['ndcg']:+.4f}, "
            f"paired interval [{ci['ndcg'][0]:+.4f}, {ci['ndcg'][1]:+.4f}]; "
            f"Recall {delta['recall']:+.4f}, "
            f"interval [{ci['recall'][0]:+.4f}, {ci['recall'][1]:+.4f}]."
        )
    text += [
        "",
        "## Failure analysis of the pre-test selected system",
        "",
        "Examples below are selected after testing for explanation, "
        "never for further model selection. "
        "Unjudged documents may also be relevant; this is analysis "
        "against the available official labels.",
        "",
    ]
    failures = sorted(
        [q for q in chosen_queries if q["recall"] < 1], key=lambda q: (q["ndcg"], q["query_id"])
    )
    for q in failures[:3]:
        text += [
            f"### Claim {q['query_id']}: {q['query']}",
            "",
            f"Recall@10 {q['recall']:.4f}; nDCG@10 {q['ndcg']:.4f}. "
            f"Missed official relevant IDs: {', '.join(q['missed_relevant_ids'])}.",
            "",
        ]
        for did in q["missed_relevant_ids"]:
            text += [f"- {did}: {dataset.corpus[did].title}"]
        if "candidate_analysis" in q:
            text += ["", "Candidate analysis: `" + json.dumps(q["candidate_analysis"]) + "`."]
        if "reranking_analysis" in q:
            text += ["", "Reranking analysis: `" + json.dumps(q["reranking_analysis"]) + "`."]
        text.append("")
    text += [
        "## Performance and reproducibility limits",
        "",
        "Every method uses the same 16 label-independent timing queries, one warmup and three "
        "measured passes (48 calls), with fresh inference. Quality uses the complete test. "
        "The CPU workstation and background load are uncontrolled; timings are descriptive. "
        "Sampled RSS covers the whole process, not isolated model weights. "
        "Preparation is separate. No GPU, other-OS, public deployment, load test, "
        "or clean-machine installation was executed. "
        "The full BEIR neural API was not executed; official BEIR files and pytrec_eval were used.",
        "",
        "All system manifests record source/model/config/data hashes, seeds, runtime versions, "
        "raw rankings, timing samples, metric definitions, and independent reference checks. "
        "Git commit identity is unavailable in this mirrored workspace. Historical development "
        "runs remain separate and are not presented as test results.",
        "",
        "## Portfolio description",
        "",
        "Built RetrievalLab, a reproducible scientific-evidence search platform comparing BM25, "
        "dense FAISS retrieval, reciprocal rank fusion, "
        "and pretrained/fine-tuned neural reranking. "
        "Implemented a shared FastAPI/Streamlit core, leakage-resistant development splits, "
        "versioned experiment evidence, held-out evaluation, and latency/memory diagnostics. "
        "Reported measured tradeoffs and regressions instead of assuming "
        "neural models always improve ranking.",
        "",
    ]
    (output / "OFFICIAL_TEST_V1.md").write_text("\n".join(text), encoding="utf-8")
    identities = {
        p.name: sha256_file(p)
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "export-manifest.json"
    }
    (output / "export-manifest.json").write_text(
        json.dumps(
            {"freeze_sha256": freeze["freeze_sha256"], "artifact_sha256": identities}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=Path("releases/v1/freeze.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/m8-official-test-v1"))
    parser.add_argument("--output", type=Path, default=Path("releases/v1/report"))
    args = parser.parse_args()
    print(json.dumps(export_release(args.freeze, args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()

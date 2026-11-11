"""Read saved validation evidence without contaminating the timed CLI process."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from retrieval_lab.data import Dataset, DatasetError, load_dataset, sha256_file
from retrieval_lab.evaluation import load_report


def render_fusion_trace(trace: dict) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Document": row["doc_id"],
                    "RRF score": row["rrf_score"],
                    "BM25 rank": row["sources"].get("bm25", {}).get("rank"),
                    "Dense rank": row["sources"].get("dense", {}).get("rank"),
                    "BM25 contribution": row["sources"].get("bm25", {}).get("contribution", 0),
                    "Dense contribution": row["sources"].get("dense", {}).get("contribution", 0),
                }
                for row in trace["results"]
            ]
        ),
        hide_index=True,
    )
    st.caption("Ranks start at 1. Outside a source's candidate list means zero contribution.")


@st.cache_data(show_spinner="Verifying the saved validation report…", max_entries=4)
def _read_report(folder: str, data_folder: str, data_hashes: str, artifact_hashes: tuple):
    dataset = load_dataset(data_folder, "train")
    if dataset.file_sha256 != json.loads(data_hashes):
        raise DatasetError("Dataset changed while loading the report")
    return load_report(Path(folder), dataset)


def render_validation(dataset: Dataset, data_folder: Path, runs_root: Path) -> None:
    st.header("Validation experiments")
    st.write(
        "Inspect measured retrieval quality, query speed, and missed evidence from a saved run."
    )
    folders = sorted(
        (
            path.parent
            for path in runs_root.glob("*/manifest.json")
            if not path.parent.name.startswith(".")
        ),
        reverse=True,
    )
    if not folders:
        st.info(
            "No validation run is saved yet. "
            "Run the experiment from the project folder, then reload."
        )
        st.code(
            "python -m retrieval_lab.cli evaluate --output runs/bm25-validation-v1", language="bash"
        )
        return
    chosen = st.selectbox("Validation run", folders, format_func=lambda path: path.name)
    try:
        names = ["manifest.json", "split.json", "queries.json", "timings.csv", "report.md"]
        hashes = tuple(sha256_file(chosen / name) for name in names)
        summary, rows, timings = _read_report(
            str(chosen), str(data_folder), json.dumps(dataset.file_sha256, sort_keys=True), hashes
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(
            "This validation report could not be verified. "
            "Choose an intact run or rerun the experiment."
        )
        st.text(str(exc))
        return
    quality, latency = summary["quality"], summary["latency"]
    st.subheader(f"{summary['system']} · {quality['query_count']} validation queries")
    st.caption(
        "Derived from official train data. These are validation results; "
        "official test labels remain reserved."
    )
    left, right = st.columns(2)
    left.metric("Recall@10", f"{quality['recall']:.4f}")
    right.metric("nDCG@10", f"{quality['ndcg']:.4f}")
    interval = summary["confidence_intervals"]
    st.caption(
        "95% group-bootstrap intervals: "
        f"Recall {interval['recall'][0]:.4f}–{interval['recall'][1]:.4f}; "
        f"nDCG {interval['ndcg'][0]:.4f}–{interval['ndcg'][1]:.4f}."
    )
    left, right = st.columns(2)
    left.metric("Query p50", f"{latency['p50_ms']:.2f} ms")
    right.metric("Query p95", f"{latency['p95_ms']:.2f} ms")
    protocol = summary["config"]["evaluation"]
    st.caption(
        f"{latency['samples']} timed calls after {protocol['warmup_passes']} full warmup pass(es); "
        f"timing covers {len(summary['measurement_protocol'].get('latency_query_ids', rows))} "
        f"of {quality['query_count']} validation queries; quality covers all queries. "
        "Full search timing includes reranker pair scoring where used. "
        "Preparation, data loading, metrics and rendering are excluded."
    )
    with st.expander("Latency distribution and process resources"):
        values = pd.Series([float(row["latency_ms"]) for row in timings])
        counts = pd.cut(values, bins=12).value_counts(sort=False)
        histogram = pd.DataFrame(
            {
                "Latency (ms)": [interval.mid for interval in counts.index],
                "Query calls": counts.to_list(),
            }
        )
        st.bar_chart(histogram, x="Latency (ms)", y="Query calls")
        st.caption(
            "Equal-width latency bins, positioned by their midpoints in milliseconds. "
            "Background workstation load was uncontrolled."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Stage": label,
                        "CPU seconds": data["process_cpu_seconds"],
                        "Sampled peak RSS (MiB)": data["sampled_peak_rss_bytes"] / 2**20,
                        "Wall seconds": data["wall_seconds"],
                    }
                    for label, data in summary["resources"].items()
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "RSS covers the whole process, including Python and loaded data. "
            "Sampling can miss short peaks. "
            "CPU/resource measurements include the runner and memory sampler."
        )
    st.divider()
    if summary.get("reranking_analysis"):
        with st.expander("Reranking input coverage and truncation"):
            analysis = summary["reranking_analysis"]
            st.write(
                f"Actual top-{analysis['rerank_depth']} input recall: "
                f"{analysis['candidate_recall']:.4f}."
            )
            st.caption(
                "This is the reranker's bounded input, not the larger BM25/dense union. "
                "Relevant documents outside this input cannot be recovered by reranking."
            )
            st.json(analysis)
    if summary.get("candidate_analysis"):
        candidate = summary["candidate_analysis"]
        with st.expander("Candidate coverage before fusion"):
            st.write(
                f"Union candidate recall: {candidate['union_recall']:.4f} at up to "
                f"{candidate['depth_per_source']} candidates per method."
            )
            st.caption(
                "This is coverage before choosing the final 10, not Recall@10. "
                "Evidence outside both lists cannot be recovered by fusion."
            )
            st.json(candidate)
    st.subheader(f"Where does {summary['system']} miss evidence?")
    st.write(
        f"{quality['queries_with_no_relevant_retrieved']} queries retrieved "
        "none of their judged relevant evidence."
    )
    selection = st.selectbox(
        "Show queries", ["Missing some judged evidence", "All validation queries"]
    )
    visible = [
        row for row in rows if selection == "All validation queries" or row["missed_relevant_ids"]
    ]
    visible.sort(key=lambda row: (row["recall"], row["ndcg"], row["query_id"]))
    if not visible:
        st.info("No queries match this filter. Select All validation queries to inspect a ranking.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": row["query_id"],
                        "Claim": row["query"],
                        "Recall@10": row["recall"],
                        "nDCG@10": row["ndcg"],
                        "Relevant found": row["relevant_retrieved"],
                        "Relevant total": row["relevant_count"],
                    }
                    for row in visible
                ]
            ),
            hide_index=True,
        )
        qid = st.selectbox(
            "Inspect a validation query",
            [row["query_id"] for row in visible],
            format_func=lambda value: f"{value} · {dataset.queries[value]}",
        )
        row = next(row for row in visible if row["query_id"] == qid)
        st.write(row["query"])
        if "reranking_trace" in row:
            with st.expander("Inspect reranking changes and candidate misses"):
                trace = row["reranking_trace"]
                st.dataframe(pd.DataFrame(trace["results"]), hide_index=True)
                st.write("Judged evidence outside the rerank input:")
                st.write(row["reranking_analysis"]["relevant_outside_rerank_input"] or "None")
                st.write("Judged input evidence lost below the final cutoff:")
                st.write(
                    row["reranking_analysis"]["relevant_in_input_below_final_cutoff"] or "None"
                )
                st.caption("Raw logits rank pairs; they are not calibrated truth probabilities.")
        if "fusion_trace" in row:
            with st.expander("Explain fusion and candidate misses"):
                render_fusion_trace(row["fusion_trace"])
                analysis = row["candidate_analysis"]
                st.write("Judged evidence outside both candidate lists:")
                st.write(analysis["relevant_missing_from_candidates"] or "None")
                st.write("Judged evidence in the candidates but outside the final top 10:")
                st.write(analysis["relevant_dropped_by_fusion"] or "None")
        st.write("**Missed judged evidence**")
        if not row["missed_relevant_ids"]:
            st.success("All judged relevant documents were retrieved in the top 10.")
        for did in row["missed_relevant_ids"]:
            doc = dataset.corpus[did]
            with st.expander(f"{doc.title} · {did}"):
                st.write(doc.text)
        st.write("**Retrieved ranking**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": hit["rank"],
                        "Document": hit["doc_id"],
                        "Title": hit["title"],
                        "Retrieval score": hit["score"],
                        "Judgment": "Unjudged"
                        if hit["judgment"] is None
                        else ("Relevant" if hit["judgment"] > 0 else "Nonrelevant"),
                    }
                    for hit in row["results"]
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "Unjudged means no label is available, not that the document is irrelevant. "
            "BM25, cosine and RRF scores use different scales."
        )
    with st.expander("Experiment identity and metric definitions"):
        st.json(
            {
                "split_sha256": summary["split_sha256"],
                "config": summary["config"],
                "metrics": summary["metric_definition"],
                "reference_check": summary["reference_check"],
                "environment": summary["environment"],
                "source_file_sha256": summary["source_file_sha256"],
                "retriever_preparation": summary.get("retriever_preparation"),
                "exact_search_check": summary.get("exact_search_check"),
            }
        )
    st.download_button(
        "Download validation summary",
        json.dumps(summary, indent=2),
        file_name="retrievallab-validation-summary.json",
        mime="application/json",
    )


@st.cache_data(show_spinner="Checking paired experiment evidence…", max_entries=4)
def _read_comparison(
    baseline: str, candidate: str, data_folder: str, data_hashes: str, artifact_hashes: tuple
):
    from retrieval_lab.comparison import compare_runs

    dataset = load_dataset(data_folder, "train")
    if dataset.file_sha256 != json.loads(data_hashes):
        raise DatasetError("Dataset changed while comparing the runs")
    return compare_runs(Path(baseline), Path(candidate), Path(data_folder))


def render_comparison(dataset: Dataset, data_folder: Path, runs_root: Path) -> None:
    st.header("Compare retrieval systems")
    st.write(
        "Compare saved runs on exactly the same validation queries. "
        "Raw BM25, cosine and RRF scores are not comparable."
    )
    folders = sorted(
        (
            path.parent
            for path in runs_root.glob("*/manifest.json")
            if not path.parent.name.startswith(".")
        ),
        reverse=True,
    )
    if len(folders) < 2:
        st.info(
            "Save at least two validation experiments to compare their quality, speed and memory."
        )
        return

    def preferred(fragment):
        return next(
            (i for i, path in enumerate(folders) if "m5" in path.name and fragment in path.name),
            next(
                (
                    i
                    for i, path in enumerate(folders)
                    if "m4" in path.name and fragment in path.name
                ),
                next((i for i, path in enumerate(folders) if fragment in path.name), 0),
            ),
        )

    with st.form("compare_runs"):
        baseline = st.selectbox(
            "Baseline run", folders, index=preferred("bm25"), format_func=lambda path: path.name
        )
        candidate = st.selectbox(
            "Candidate run",
            folders,
            index=preferred("rerank")
            if any("rerank" in p.name for p in folders)
            else preferred("hybrid")
            if any("hybrid" in p.name for p in folders)
            else preferred("dense"),
            format_func=lambda path: path.name,
        )
        submitted = st.form_submit_button("Compare runs", type="primary")
    try:
        names = ["manifest.json", "split.json", "queries.json", "timings.csv", "report.md"]
        hashes = tuple(
            sha256_file(folder / name) for folder in [baseline, candidate] for name in names
        )
        signature = (
            str(baseline),
            str(candidate),
            hashes,
            json.dumps(dataset.file_sha256, sort_keys=True),
        )
        if submitted:
            result = _read_comparison(
                str(baseline), str(candidate), str(data_folder), signature[3], hashes
            )
            st.session_state["comparison"] = (signature, result)
        saved = st.session_state.get("comparison")
        if saved is None or saved[0] != signature:
            st.info(
                "Select two runs and compare. "
                "Mismatched splits or evaluation settings are rejected."
            )
            return
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"These experiments could not be compared: {exc}")
        return
    report = saved[1]
    st.subheader(f"Paired comparison · {report['query_count']} queries")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Role": row["role"],
                    "System": row["system"],
                    "Recall@10": row["recall_at_10"],
                    "nDCG@10": row["ndcg_at_10"],
                    "p50 (ms)": row["p50_ms"],
                    "p95 (ms)": row["p95_ms"],
                    "Process peak RSS (MiB, sampled)": row["query_stage_sampled_peak_rss_mib"],
                    "Preparation (s)": row["preparation_seconds"],
                }
                for row in report["systems"]
            ]
        ),
        hide_index=True,
    )
    delta, interval = report["candidate_minus_baseline"], report["paired_95pct_intervals"]
    st.write(f"Candidate minus baseline: Recall {delta['recall']:+.4f}; nDCG {delta['ndcg']:+.4f}.")
    st.caption(
        f"Paired 95% group-bootstrap intervals: Recall [{interval['recall'][0]:+.4f}, "
        f"{interval['recall'][1]:+.4f}]; nDCG "
        f"[{interval['ndcg'][0]:+.4f}, {interval['ndcg'][1]:+.4f}]."
    )
    st.caption(report["timing_caveat"])
    if not report["same_environment_snapshot"]:
        st.warning(
            "The environment snapshots differ. "
            "Read the individual manifests before interpreting resource differences."
        )
    st.subheader("Which queries changed?")
    outcomes = sorted(
        report["per_query"], key=lambda row: (-abs(row["ndcg_delta"]), row["query_id"])
    )
    st.dataframe(pd.DataFrame(outcomes), hide_index=True)
    with st.expander("Coverage and query-level outcomes"):
        st.json(
            {
                "coverage": report["coverage"],
                "outcomes": report["per_query_outcomes"],
                "source_runs": report["source_runs"],
                "split_sha256": report["split_sha256"],
            }
        )
    st.download_button(
        "Download paired comparison",
        json.dumps(report, indent=2),
        file_name="retrievallab-comparison.json",
        mime="application/json",
    )
    from retrieval_lab.exports import comparison_csv, comparison_figure

    st.download_button(
        "Download comparison CSV",
        comparison_csv(report),
        file_name="retrievallab-comparison.csv",
        mime="text/csv",
    )
    try:
        figure = comparison_figure(report)
        st.image(figure, caption="Quality, response time and memory have different units.")
        st.download_button(
            "Download comparison figure",
            figure,
            file_name="retrievallab-comparison.png",
            mime="image/png",
        )
    except ImportError:
        st.info("Install the platform extra to enable downloadable comparison figures.")

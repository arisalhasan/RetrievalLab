"""Read-only final results; neither training nor evaluation runs inside Streamlit."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from retrieval_lab.data import load_dataset
from retrieval_lab.evaluation import load_report


def render_final(root: Path, data_dir: Path):
    st.header("The held-out test")
    st.write(
        "Settings are chosen on validation data, then locked before the official test. "
        "Test results describe generalization; they do not select the released model."
    )
    freeze_path = root / "releases/v1/freeze.json"
    output = root / "runs/m8-official-test-v1"
    if not freeze_path.exists() or not (output / "result.json").exists():
        st.info(
            "Final test results will appear after the frozen evaluation finishes. "
            "Search and validation analysis are available in the other tabs."
        )
        return
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        dataset = load_dataset(data_dir, "test")
        reports, rows = {}, []
        for name in freeze["systems"]:
            summary, queries, _ = load_report(output / name, dataset)
            if summary.get("freeze_sha256") != freeze["freeze_sha256"]:
                raise ValueError("Final result does not match the release freeze")
            reports[name] = (summary, queries)
            rows.append(
                {
                    "System": name,
                    "Recall@10": summary["quality"]["recall"],
                    "nDCG@10": summary["quality"]["ndcg"],
                    "p95 (ms)": summary["latency"]["p95_ms"],
                    "Missed queries": summary["quality"]["queries_with_no_relevant_retrieved"],
                }
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Final report verification failed: {exc}")
        return
    st.subheader(f"Selected before test: {freeze['selected_system']}")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "Quality covers every official test query. Timing uses a fixed sample. "
        "Measurements come from one CPU workstation; memory is whole-process sampled RSS."
    )
    st.download_button(
        "Download final scores CSV",
        pd.DataFrame(rows).to_csv(index=False),
        file_name="retrievallab-official-test.csv",
        mime="text/csv",
    )
    st.subheader("Inspect a failure")
    system = st.selectbox("Final system", list(reports), key="final_system")
    summary, queries = reports[system]
    failures = sorted(
        [q for q in queries if q["recall"] < 1], key=lambda q: (q["ndcg"], q["query_id"])
    )
    if failures:
        row = st.selectbox(
            "Incomplete evidence retrieval",
            failures,
            format_func=lambda q: f"{q['query_id']} · {q['query']}",
        )
        st.write(f"Recall@10 {row['recall']:.3f} · nDCG@10 {row['ndcg']:.3f}")
        st.dataframe(pd.DataFrame(row["results"]), hide_index=True)
        for did in row["missed_relevant_ids"]:
            with st.expander(f"Missed relevant document · {did}"):
                st.write(dataset.corpus[did].title)
                st.write(dataset.corpus[did].text)
        st.caption(
            "Post-test error analysis only. These examples must not drive new tuning "
            "followed by claims of an untouched test set."
        )
    with st.expander("Release identity and measurement protocol"):
        st.json(
            {
                "freeze_sha256": freeze["freeze_sha256"],
                "selection": freeze["selection_rule"],
                "protocol": summary["measurement_protocol"],
                "uncertainty": summary["confidence_intervals"],
            }
        )

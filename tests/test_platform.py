import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retrieval_lab.api import SearchService, create_app
from retrieval_lab.exports import comparison_csv, comparison_figure

ROOT = Path(__file__).resolve().parents[1]


def test_api_real_bm25_and_no_test_labels(mini_dataset):
    (mini_dataset / "qrels/test.tsv").write_text("NOT FOR API", encoding="utf-8")
    service = SearchService(ROOT, mini_dataset)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["loaded_systems"] == []
        response = client.post("/search", json={"query": "planets", "top_k": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["doc_id"] == "d2"
        assert data["run_type"] == "interactive_search_not_benchmark"
        assert data["query_time_ms"] >= 0
        assert client.get("/health").json()["loaded_systems"] == ["bm25"]
        assert "trained" in client.get("/systems").json()["systems"]
        assert "SearchResponse" in client.get("/openapi.json").json()["components"]["schemas"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "x" * 2001},
        {"query": "x", "top_k": 21},
        {"query": "x", "top_k": True},
        {"query": "x", "system": "unknown"},
        {"query": "x", "oops": 1},
        {},
    ],
)
def test_api_rejects_invalid_requests(payload):
    with TestClient(create_app()) as client:
        assert client.post("/search", json=payload).status_code == 422


def test_api_missing_assets_and_busy(tmp_path):
    service = SearchService(tmp_path)
    with TestClient(create_app(service)) as client:
        result = client.post("/search", json={"query": "cells"})
        assert result.status_code == 503
        assert str(tmp_path) not in result.text
        service.gate.acquire()
        try:
            result = client.post("/search", json={"query": "cells"})
            assert result.status_code == 429
            assert result.headers["retry-after"] == "1"
        finally:
            service.gate.release()


def test_exports_preserve_measured_values():
    # Tiny synthetic export values; offline tests must not require generated benchmark runs.
    report = {
        "query_count": 2,
        "systems": [
            {
                "role": role,
                "system": "Synthetic fixture",
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
                "p95_ms": latency,
                "query_stage_sampled_peak_rss_mib": memory,
            }
            for role, recall, ndcg, latency, memory in [
                ("baseline", 0.8, 0.7, 10.0, 100.0),
                ("candidate", 0.9, 0.75, 20.0, 150.0),
            ]
        ],
    }
    table = pd.read_csv(io.BytesIO(comparison_csv(report)))
    assert table["ndcg_at_10"].tolist() == pytest.approx(
        [r["ndcg_at_10"] for r in report["systems"]]
    )
    assert comparison_figure(report).startswith(b"\x89PNG")
    assert b"<svg" in comparison_figure(report, "svg")
    with pytest.raises(ValueError):
        comparison_figure(report, "pdf")

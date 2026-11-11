import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from retrieval_lab.bm25 import BM25Config
from retrieval_lab.hybrid import HybridConfig, HybridIndex, fuse_rankings
from retrieval_lab.types import Document, SearchResult


def hits(ids, scale=1.0):
    return [
        SearchResult(Document(did, did, "abstract"), scale / (i + 1)) for i, did in enumerate(ids)
    ]


def test_hand_calculation_consensus_one_based_ranks_and_missing_sources():
    results, trace = fuse_rankings({"bm25": hits(["a", "b"]), "dense": hits(["c", "b"])})
    assert [hit.document.doc_id for hit in results] == ["b", "a", "c"]
    assert [hit.score for hit in results] == pytest.approx([2 / 62, 1 / 61, 1 / 61])
    assert trace["union_size"] == 3
    assert trace["results"][0]["sources"]["bm25"] == {"rank": 2, "contribution": 1 / 62}
    assert "dense" not in trace["results"][1]["sources"]
    assert json.loads(json.dumps(trace)) == trace


def test_scores_do_not_matter_and_input_source_order_does_not_break_ties():
    expected = fuse_rankings({"lexical": hits(["b", "a"]), "neural": hits(["a", "b"])})[0]
    actual = fuse_rankings({"neural": hits(["a", "b"], -1000), "lexical": hits(["b", "a"], 1e9)})[0]
    assert actual == expected
    assert [hit.document.doc_id for hit in actual] == ["a", "b"]


def test_against_independent_exact_rational_rank_table():
    rng = random.Random(42)
    ids = [str(i) for i in range(40)]
    rankings = {name: hits(rng.sample(ids, 25)) for name in ["a", "b", "c"]}
    actual = fuse_rankings(rankings, 60, 12)[0]
    reference = {}
    for did in ids:
        terms = [
            Fraction(1, 60 + ids_for_source.index(did) + 1)
            for ranking in rankings.values()
            if did in (ids_for_source := [hit.document.doc_id for hit in ranking])
        ]
        if terms:
            reference[did] = sum(terms)
    expected = sorted(reference, key=lambda did: (-reference[did], did))[:12]
    assert [hit.document.doc_id for hit in actual] == expected
    assert [hit.score for hit in actual] == pytest.approx([float(reference[d]) for d in expected])


def test_empty_sources_short_lists_and_zero_constant():
    assert fuse_rankings({"empty": []})[0] == []
    assert fuse_rankings({})[1]["union_size"] == 0
    result, _ = fuse_rankings({"a": [], "b": hits(["x"])}, rank_constant=0, top_k=20)
    assert result[0].score == 1


def test_component_duplicate_ids_and_conflicting_payloads_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        fuse_rankings({"a": hits(["x", "x"])})
    with pytest.raises(ValueError, match="Conflicting"):
        fuse_rankings({"a": hits(["x"]), "b": [SearchResult(Document("x", "other", ""), 1)]})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank_constant": -1},
        {"rank_constant": True},
        {"rank_constant": 0.5},
        {"candidate_depth": 0},
        {"candidate_depth": True},
        {"system": "dense"},
        {"bm25": {}},
        {"schema_version": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_config_snapshots_both_components_and_round_trips(tmp_path):
    config = HybridConfig(bm25=BM25Config(k1=1.5))
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert HybridConfig.from_file(path) == config
    root = Path(__file__).resolve().parents[1]
    assert HybridConfig.from_file(root / "configs/hybrid.json") == HybridConfig()
    payload = config.to_dict()
    payload["dense"]["revision"] = "main"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        HybridConfig.from_file(path)


def test_hybrid_calls_live_children_once_per_search_and_enforces_depth():
    class Component:
        def __init__(self, ids):
            self.ids, self.calls = ids, []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return hits(self.ids[:top_k])

    bm25, dense = Component(["a", "b"]), Component(["c", "b"])
    index = HybridIndex(bm25, dense, HybridConfig(candidate_depth=2))
    first, trace = index.search_with_trace("query", 1)
    assert index.search("query", 1) == first
    assert first[0].document.doc_id == "b"
    assert trace["source_candidate_ids"] == {"bm25": ["a", "b"], "dense": ["c", "b"]}
    assert bm25.calls == dense.calls == [("query", 2), ("query", 2)]
    with pytest.raises(ValueError, match="candidate_depth"):
        index.search("query", 3)
    with pytest.raises(ValueError):
        index.search("query", True)


def test_candidate_misses_fusion_losses_no_positive_and_self_id_are_distinct():
    from retrieval_lab.evaluation import analyze_candidates, summarize_candidates

    trace = {"source_candidate_ids": {"bm25": ["q", "a", "b"], "dense": ["b", "c"]}}
    row = analyze_candidates(trace, "q", {"a": 1, "b": 1, "missing": 1, "unjudged": 0}, ["b"])
    assert row["union_recall"] == 2 / 3
    assert row["union_size_after_self_filter"] == 3
    assert row["relevant_missing_from_candidates"] == ["missing"]
    assert row["relevant_dropped_by_fusion"] == ["a"]
    assert row["source_recall"] == {"bm25": 2 / 3, "dense": 1 / 3}
    empty = analyze_candidates(trace, "q", {}, [])
    assert empty["union_recall"] == 0
    summary = summarize_candidates([row, empty], 100)
    assert summary["union_recall"] == 1 / 3
    assert summary["queries_with_evidence_missing_from_candidates"] == 1
    assert summary["queries_with_candidate_evidence_dropped_from_top10"] == 1

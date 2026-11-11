import json
import os
from pathlib import Path

import pytest

from retrieval_lab.hybrid import HybridConfig
from retrieval_lab.reranking import RerankedIndex, RerankerConfig, rerank_candidates
from retrieval_lab.types import Document, SearchResult


def candidates():
    return [
        SearchResult(Document(did, did, "abstract"), score)
        for did, score in [("b", 100), ("a", 5), ("c", 1)]
    ]


def test_rerank_uses_joint_scores_only_preserves_membership_and_breaks_ties_by_id():
    hits, trace = rerank_candidates(candidates(), [-4, 3, 3], 2)
    assert [hit.document.doc_id for hit in hits] == ["a", "c"]
    assert [hit.score for hit in hits] == [3, 3]
    assert trace["results"][0]["retrieval_rank"] == 2
    assert trace["candidates"][0]["retrieval_score"] == 100
    assert json.loads(json.dumps(trace)) == trace
    assert rerank_candidates([], [], 10)[0] == []


@pytest.mark.parametrize(
    "scores,k", [([1], 10), ([1, float("nan"), 2], 10), ([1, float("inf"), 2], 10), ([1, 2, 3], 0)]
)
def test_bad_scores_and_cutoff_are_rejected(scores, k):
    with pytest.raises(ValueError):
        rerank_candidates(candidates(), scores, k)


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        rerank_candidates([candidates()[0], candidates()[0]], [1, 2], 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revision": "main"},
        {"rerank_depth": 0},
        {"rerank_depth": 101},
        {"max_length": 513},
        {"max_length": True},
        {"batch_size": 0},
        {"device": "cuda"},
        {"activation": "sigmoid"},
        {"seed": True},
        {"retriever": {}},
        {"torch_threads": 2},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RerankerConfig(**kwargs)


def test_nested_config_round_trip_and_fixed_baseline(tmp_path):
    config = RerankerConfig(retriever=HybridConfig(candidate_depth=75), rerank_depth=25)
    path = tmp_path / "rerank.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert RerankerConfig.from_file(path) == config
    root = Path(__file__).resolve().parents[1]
    assert RerankerConfig.from_file(root / "configs/reranker.json") == RerankerConfig()


class SyntheticPairScorer:
    fingerprint = {"backend": "synthetic-unit-test-pair-scores"}

    def __init__(self):
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [float("planet" in document.casefold()) for query, document in pairs]

    def token_statistics(self, pairs):
        return {"pairs": len(pairs), "pairs_truncated": 0, "synthetic": True}


def test_live_retrieval_and_pair_scoring_on_every_call_without_query_cache():
    class Retrieval:
        preparation = {"synthetic": True}

        def __init__(self):
            self.calls = []

        def search_with_trace(self, query, top_k):
            self.calls.append((query, top_k))
            return candidates(), {"synthetic": True}

    retriever, scorer = Retrieval(), SyntheticPairScorer()
    index = RerankedIndex(retriever, scorer, RerankerConfig())
    assert index.search("query", 2) == index.search("query", 2)
    assert retriever.calls == [("query", 50), ("query", 50)]
    assert len(scorer.calls) == 2 and len(scorer.calls[0]) == 3
    assert scorer.calls[0][0] == ("query", "b\nabstract")
    with pytest.raises(ValueError, match="rerank_depth"):
        index.search("query", 51)


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("RUN_RERANKER_MODEL_TESTS") != "1", reason="Opt-in real reranker"
)
def test_real_tinybert_scores_match_direct_logits_and_repeat():
    import numpy as np
    import torch

    from retrieval_lab.cross_encoder import CrossEncoderScorer

    root = Path(__file__).resolve().parents[1]
    scorer = CrossEncoderScorer(RerankerConfig(), root / "models")
    pairs = [
        ("Which planet is red?", "Mars is known as the red planet."),
        ("Which planet is red?", "Mitosis is a form of cell division."),
    ]
    scores = scorer.score(pairs)
    assert np.isfinite(scores).all() and scores[0] > scores[1]
    assert scores == scorer.score(pairs)
    batch = scorer.model.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        direct = scorer.model.model(**batch).logits[:, 0].numpy()
    assert scores == pytest.approx(direct.tolist(), abs=1e-6)
    assert scorer.token_statistics(pairs)["pairs_truncated"] == 0
    assert scorer.token_statistics([("query", "word " * 1000)])["pairs_truncated"] == 1

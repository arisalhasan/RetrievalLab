"""Recall and linear-gain nDCG for an explicit ordered ranking."""

import math
from statistics import fmean

METRIC_VERSION = "trec-linear-gain-v1"


def score_query(
    query_id: str, ranked_ids: list[str], judgments: dict[str, int], k: int = 10
) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Metric cutoff must be a positive integer")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Rankings must not contain duplicate document IDs")
    if any(
        isinstance(grade, bool) or not isinstance(grade, int) or grade < 0
        for grade in judgments.values()
    ):
        raise ValueError("Relevance grades must be nonnegative integers")
    # BEIR defaults to removing documents whose ID equals the query ID.
    ranking = [did for did in ranked_ids if did != query_id][:k]
    relevant = {did for did, grade in judgments.items() if grade > 0}
    gains = [judgments.get(did, 0) for did in ranking]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(sorted(judgments.values(), reverse=True)[:k], 1)
    )
    found = len(set(ranking) & relevant)
    return {
        "recall": found / len(relevant) if relevant else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
        "relevant_count": len(relevant),
        "relevant_retrieved": found,
        "retrieved_count": len(ranking),
        "first_relevant_rank": next(
            (i for i, did in enumerate(ranking, 1) if did in relevant), None
        ),
    }


def aggregate(scores: dict[str, dict]) -> dict:
    if not scores:
        raise ValueError("Cannot evaluate an empty query set")
    return {
        "query_count": len(scores),
        "recall": fmean(row["recall"] for row in scores.values()),
        "ndcg": fmean(row["ndcg"] for row in scores.values()),
        "queries_without_positive_labels": sum(
            row["relevant_count"] == 0 for row in scores.values()
        ),
        "queries_with_no_relevant_retrieved": sum(
            row["relevant_count"] > 0 and row["relevant_retrieved"] == 0 for row in scores.values()
        ),
        "zero_result_queries": sum(row["retrieved_count"] == 0 for row in scores.values()),
    }


def check_reference(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
    scores: dict[str, dict],
    k: int = 10,
) -> dict:
    """Cross-check the same ordering with pytrec_eval, the backend used by BEIR.

    Rank-based surrogate scores preserve our tie policy. Raw BM25 scores remain
    in run artifacts; this transformation is for reference metric calculation only.
    """
    import pytrec_eval

    reference_runs = {
        qid: {did: float(len(ranking) - rank) for rank, did in enumerate(ranking) if did != qid}
        for qid, ranking in rankings.items()
    }
    nonempty_qrels = {qid: rels for qid, rels in qrels.items() if rels}
    evaluator = pytrec_eval.RelevanceEvaluator(nonempty_qrels, {f"recall.{k}", f"ndcg_cut.{k}"})
    reference = evaluator.evaluate(reference_runs)
    max_error = 0.0
    for qid, observed in scores.items():
        if qid not in reference and reference_runs[qid] and qrels[qid]:
            raise ValueError(f"Reference evaluator omitted nonempty query {qid}")
        expected = reference.get(qid, {f"recall_{k}": 0.0, f"ndcg_cut_{k}": 0.0})
        for ours, theirs in [("recall", f"recall_{k}"), ("ndcg", f"ndcg_cut_{k}")]:
            error = abs(observed[ours] - expected[theirs])
            max_error = max(max_error, error)
            if not math.isclose(observed[ours], expected[theirs], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Metric disagrees with pytrec_eval for query {qid}: {ours}")
    return {
        "backend": "pytrec_eval (Terrier distribution)",
        "passed": True,
        "query_count": len(scores),
        "max_absolute_error": max_error,
        "ordering": "rank-preserving surrogate scores; self IDs removed",
        "empty_judgments_policy": "zero, included in our macro average",
    }

import math

import pytest

from retrieval_lab.metrics import aggregate, check_reference, score_query


def test_hand_calculated_recall_and_discounted_ranking():
    row = score_query("q", ["wrong", "a", "b"], {"a": 1, "b": 1, "c": 1}, k=3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["ndcg"] == pytest.approx((1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3) + 1 / 2))
    assert row["first_relevant_rank"] == 2


def test_linear_graded_gain_not_exponential():
    assert score_query("q", ["low", "high"], {"high": 3, "low": 1}, 2)["ndcg"] == pytest.approx(
        (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    )


def test_cutoff_short_rankings_unjudged_and_self_id():
    row = score_query("q", ["q", "unjudged", "a", "b"], {"q": 1, "a": 1, "b": 1}, 2)
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["retrieved_count"] == 2
    assert score_query("q", ["a"], {"a": 1})["ndcg"] == 1
    assert score_query("q", ["a"], {"a": 1})["recall"] == 1


def test_empty_and_no_positive_queries_remain_in_macro_average():
    scores = {
        "a": score_query("a", ["d"], {"d": 1}),
        "b": score_query("b", [], {"e": 1}),
        "c": score_query("c", ["x"], {"x": 0}),
    }
    summary = aggregate(scores)
    assert summary["recall"] == pytest.approx(1 / 3)
    assert summary["queries_without_positive_labels"] == 1
    assert summary["zero_result_queries"] == 1
    assert summary["queries_with_no_relevant_retrieved"] == 1


@pytest.mark.parametrize(
    "rankings,judgments,k",
    [
        (["d", "d"], {"d": 1}, 10),
        (["d"], {"d": -1}, 10),
        (["d"], {"d": 0.5}, 10),
        (["d"], {"d": 1}, 0),
    ],
)
def test_invalid_metric_inputs(rankings, judgments, k):
    with pytest.raises(ValueError):
        score_query("q", rankings, judgments, k)


def test_metrics_match_reference_on_short_empty_graded_and_self_rankings():
    qrels = {
        "q1": {"a": 1, "b": 1},
        "q2": {"a": 1},
        "q3": {"c": 0},
        "q4": {"high": 3, "low": 1},
        "q5": {"q5": 1, "a": 1},
    }
    rankings = {
        "q1": ["x", "a", "b"],
        "q2": [],
        "q3": ["c"],
        "q4": ["low", "high"],
        "q5": ["q5", "a"],
    }
    scores = {qid: score_query(qid, rankings[qid], rels) for qid, rels in qrels.items()}
    assert check_reference(qrels, rankings, scores)["passed"]

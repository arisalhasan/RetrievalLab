import math

import pytest

from retrieval_lab.bm25 import BM25Config, BM25Index, tokenize
from retrieval_lab.types import Document


def test_tokenizer_handles_case_punctuation_unicode_and_numbers():
    assert tokenize("IL-6, naïve CELLS_2027!") == ["il", "6", "naïve", "cells", "2027"]


def test_score_matches_hand_calculated_bm25():
    index = BM25Index([Document("a", "", "cat cat"), Document("b", "", "dog")])
    idf = math.log1p(1.5 / 1.5)
    expected = idf * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
    (hit,) = index.search("cat")
    assert hit.document.doc_id == "a"
    assert hit.score == pytest.approx(expected)


def test_titles_are_indexed_and_query_repetition_is_binary():
    index = BM25Index([Document("a", "Astronomy", "A planet.")])
    assert index.search("astronomy") == index.search("ASTRONOMY astronomy")


def test_deterministic_ties_independent_of_input_order():
    docs = [Document("b", "", "cell"), Document("a", "", "cell")]
    assert [r.document.doc_id for r in BM25Index(docs).search("cell")] == ["a", "b"]
    assert BM25Index(docs).search("cell") == BM25Index(docs[::-1]).search("cell")


@pytest.mark.parametrize("query", ["", "  ", "!!!", "unknown"])
def test_queries_without_matches_return_no_arbitrary_documents(query):
    assert BM25Index([Document("a", "", "cell")]).search(query) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        BM25Index([Document("a", "", "cell")]).search("cell", top_k)


def test_top_k_and_high_document_frequency():
    index = BM25Index([Document(str(i), "", "cell") for i in range(3)])
    assert len(index.search("cell", 2)) == 2
    assert all(hit.score > 0 for hit in index.search("cell", 10))


@pytest.mark.parametrize(
    "docs", [[], [Document("a", "", "!!!")], [Document("a", "", "x"), Document("a", "", "y")]]
)
def test_rejects_invalid_corpora(docs):
    with pytest.raises(ValueError):
        BM25Index(docs)


@pytest.mark.parametrize(
    "kwargs", [{"k1": 0}, {"k1": float("nan")}, {"b": -0.1}, {"b": 1.1}, {"b": float("inf")}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        BM25Config(**kwargs)

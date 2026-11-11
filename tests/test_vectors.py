import importlib.util

import numpy as np
import pytest

from retrieval_lab.types import Document
from retrieval_lab.vectors import ExactVectorIndex, unit_vectors

requires_faiss = pytest.mark.skipif(
    importlib.util.find_spec("faiss") is None, reason="Install the dense optional dependencies"
)


@requires_faiss
def test_faiss_matches_independent_cosine_and_canonical_ties():
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [Document("b", "", "b"), Document("a", "", "a"), Document("c", "", "c")]
    vectors = np.array([[2, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32)
    index = ExactVectorIndex(docs, vectors)
    hits = index.search_vector(np.array([1, 0, 0]), 3)
    assert [hit.document.doc_id for hit in hits] == ["a", "b", "c"]
    expected = cosine_similarity(np.array([[1, 0, 0]], dtype=np.float32), index.embeddings)[0]
    assert [hit.score for hit in hits] == pytest.approx(expected.tolist(), abs=1e-6)
    assert index.verify_exact(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))["passed"]


@pytest.mark.parametrize(
    "values", [np.zeros((2, 3)), np.ones(3), np.array([[np.nan, 1]]), np.empty((0, 3))]
)
def test_bad_embedding_matrices(values):
    with pytest.raises(ValueError):
        unit_vectors(values)


@requires_faiss
def test_query_shape_k_and_corpus_alignment_are_checked():
    doc = Document("a", "", "text")
    index = ExactVectorIndex([doc], np.array([[1, 0]], dtype=np.float32))
    for vector, k in [(np.ones(3), 10), (np.ones(2), 0)]:
        with pytest.raises(ValueError):
            index.search_vector(vector, k)
    with pytest.raises(ValueError):
        ExactVectorIndex([doc, doc], np.ones((2, 2)))
    with pytest.raises(ValueError):
        ExactVectorIndex([doc], np.ones((2, 2)))

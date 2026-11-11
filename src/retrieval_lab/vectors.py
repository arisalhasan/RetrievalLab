"""Exact cosine retrieval with FAISS and an independent NumPy oracle."""

import numpy as np

from retrieval_lab.types import Document, SearchResult


def unit_vectors(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Embeddings must be a nonempty 2D matrix")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-length embeddings cannot define cosine similarity")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class ExactVectorIndex:
    """Flat inner product over normalized vectors, with canonical ID tie breaking.

    At SciFact scale, requesting all scores makes boundary ties deterministic.
    This is an exact reference baseline, not a scalable approximate index.
    """

    def __init__(self, documents: list[Document], embeddings: np.ndarray, threads: int = 1):
        import faiss

        if not documents or len({doc.doc_id for doc in documents}) != len(documents):
            raise ValueError("Vector documents must be nonempty with unique IDs")
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding row counts differ")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("FAISS threads must be a positive integer")
        order = sorted(range(len(documents)), key=lambda i: documents[i].doc_id)
        self.documents = [documents[i] for i in order]
        self.embeddings = unit_vectors(embeddings)[order].copy()
        self.dimension = self.embeddings.shape[1]
        self.threads = threads
        faiss.omp_set_num_threads(threads)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

    def _query(self, vector: np.ndarray, top_k: int) -> np.ndarray:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,):
            raise ValueError("Query embedding dimension differs from the index")
        return unit_vectors(vector[None, :])

    def search_vector(self, vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        import faiss

        vector = self._query(vector, top_k)
        # FAISS thread settings are thread-local on some builds (including Streamlit workers).
        faiss.omp_set_num_threads(self.threads)
        scores, labels = self.index.search(vector, len(self.documents))
        order = np.lexsort((labels[0], -scores[0]))[:top_k]
        return [SearchResult(self.documents[int(labels[0, i])], float(scores[0, i])) for i in order]

    def numpy_search(self, vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        vector = self._query(vector, top_k)[0]
        scores = self.embeddings @ vector
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
        return [SearchResult(self.documents[int(i)], float(scores[i])) for i in order]

    def verify_exact(self, vectors: np.ndarray, top_k: int = 10, atol: float = 1e-6) -> dict:
        strict_agreements, max_error, numerical_ties = 0, 0.0, 0
        for vector in vectors:
            faiss_hits = self.search_vector(vector, top_k)
            oracle_hits = self.numpy_search(vector, top_k)
            a = [hit.document.doc_id for hit in faiss_hits]
            b = [hit.document.doc_id for hit in oracle_hits]
            if a == b:
                strict_agreements += 1
            else:
                # Float32 kernels can reorder numerically indistinguishable neighbors.
                oracle_all = self.embeddings @ self._query(vector, top_k)[0]
                positions = {doc.doc_id: i for i, doc in enumerate(self.documents)}
                selected = [float(oracle_all[positions[did]]) for did in a]
                expected = [hit.score for hit in oracle_hits]
                if not np.allclose(selected, expected, atol=atol, rtol=0):
                    raise ValueError("FAISS ranking disagrees with exact NumPy cosine search")
                numerical_ties += 1
            oracle_all = self.embeddings @ self._query(vector, top_k)[0]
            positions = {doc.doc_id: i for i, doc in enumerate(self.documents)}
            errors = [
                abs(hit.score - float(oracle_all[positions[hit.document.doc_id]]))
                for hit in faiss_hits
            ]
            error = max(errors, default=0.0)
            max_error = max(max_error, error)
            if error > atol:
                raise ValueError("FAISS cosine scores disagree with the NumPy oracle")
        return {
            "passed": True,
            "queries": len(vectors),
            "strict_rank_agreements": strict_agreements,
            "float32_tie_reorderings": numerical_ties,
            "max_absolute_score_error": max_error,
            "absolute_tolerance": atol,
            "oracle": "independent full NumPy dot product",
        }

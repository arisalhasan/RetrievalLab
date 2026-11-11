"""Versioned embedding persistence and text-query retrieval; labels never enter."""

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from retrieval_lab.data import DatasetError, sha256_file
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.files import install_directory
from retrieval_lab.splits import canonical_hash
from retrieval_lab.types import Document, SearchResult
from retrieval_lab.vectors import ExactVectorIndex, unit_vectors


class Encoder(Protocol):
    fingerprint: dict

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def token_statistics(self, texts: list[str]) -> dict: ...


def corpus_fingerprint(documents: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda item: item.doc_id):
        digest.update(
            (json.dumps([doc.doc_id, doc.title, doc.text], ensure_ascii=False) + "\n").encode()
        )
    return digest.hexdigest()


def embedding_identity(documents: list[Document], config: DenseConfig, encoder: Encoder) -> dict:
    root = Path(__file__).parent
    return {
        "schema_version": 1,
        "corpus_sha256": corpus_fingerprint(documents),
        "config": config.to_dict(),
        "encoder": encoder.fingerprint,
        "implementation_sha256": {
            name: sha256_file(root / name) for name in ["dense.py", "encoders.py", "vectors.py"]
        },
    }


def prepare_embeddings(
    documents: list[Document],
    config: DenseConfig,
    encoder: Encoder,
    cache_root: Path,
    *,
    allow_build: bool = True,
) -> tuple[np.ndarray, dict]:
    documents = sorted(documents, key=lambda doc: doc.doc_id)
    if not documents or len({doc.doc_id for doc in documents}) != len(documents):
        raise ValueError("Embedding corpus must be nonempty with unique IDs")
    identity = embedding_identity(documents, config, encoder)
    key = canonical_hash(identity)
    target = cache_root / key
    ids = [doc.doc_id for doc in documents]
    started = perf_counter()
    if target.exists():
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if manifest["identity"] != identity:
                raise DatasetError("Embedding cache identity does not match this experiment")
            for name in ["embeddings.npy", "doc_ids.json"]:
                if sha256_file(target / name) != manifest["file_sha256"][name]:
                    raise DatasetError(f"Embedding cache checksum mismatch: {name}")
            if json.loads((target / "doc_ids.json").read_text(encoding="utf-8")) != ids:
                raise DatasetError("Embedding rows do not match canonical document IDs")
            vectors = np.load(target / "embeddings.npy", allow_pickle=False)
            if vectors.dtype != np.float32 or vectors.shape != (len(ids), config.dimension):
                raise DatasetError("Embedding cache has unexpected dtype or shape")
            if (
                not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                or not np.isfinite(vectors).all()
            ):
                raise DatasetError("Embedding cache is not finite unit vectors")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError(
                f"Cannot reuse the embedding cache: {exc}. Use a new --cache-root."
            ) from exc
        return vectors, {
            "cache_hit": True,
            "cache_key": key,
            "embedding_bytes": vectors.nbytes,
            "cache_load_seconds": perf_counter() - started,
            "original_preparation": manifest["preparation"],
            "identity": identity,
        }
    if not allow_build:
        raise DatasetError("No matching embedding cache. Run: retrievallab build-dense")
    texts = [f"{doc.title}\n{doc.text}" for doc in documents]
    token_stats = encoder.token_statistics(texts)
    encoding_started = perf_counter()
    vectors = unit_vectors(encoder.encode(texts))
    encoding_seconds = perf_counter() - encoding_started
    if vectors.shape != (len(ids), config.dimension):
        raise ValueError("Encoder returned an unexpected document embedding shape")
    preparation = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_encoding_seconds": encoding_seconds,
        "token_statistics": token_stats,
        "documents": len(documents),
        "dimension": config.dimension,
        "embedding_bytes": vectors.nbytes,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".embeddings-", dir=cache_root) as temp:
        stage = Path(temp) / "index"
        stage.mkdir()
        np.save(stage / "embeddings.npy", vectors, allow_pickle=False)
        (stage / "doc_ids.json").write_text(json.dumps(ids) + "\n", encoding="utf-8")
        manifest = {
            "identity": identity,
            "preparation": preparation,
            "file_sha256": {
                name: sha256_file(stage / name) for name in ["embeddings.npy", "doc_ids.json"]
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        install_directory(stage, target, cache_root)
    return vectors, {
        "cache_hit": False,
        "cache_key": key,
        "embedding_bytes": vectors.nbytes,
        "cache_load_seconds": None,
        "original_preparation": preparation,
        "identity": identity,
    }


class DenseIndex:
    def __init__(
        self,
        documents: list[Document],
        config: DenseConfig,
        encoder: Encoder,
        cache_root: Path,
        *,
        allow_build: bool = True,
    ):
        self.config, self.encoder = config, encoder
        ordered = sorted(documents, key=lambda doc: doc.doc_id)
        vectors, self.preparation = prepare_embeddings(
            ordered, config, encoder, cache_root, allow_build=allow_build
        )
        started = perf_counter()
        self.vector_index = ExactVectorIndex(ordered, vectors, config.faiss_threads)
        self.preparation["faiss_build_seconds"] = perf_counter() - started

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not any(char.isalnum() for char in query):
            return []
        vector = self.encoder.encode([query])[0]
        return self.vector_index.search_vector(vector, top_k)

    def sanity_check(self, queries: list[str], top_k: int = 10) -> dict:
        # Untimed and uncached. Measured searches still encode every query independently.
        vectors = self.encoder.encode(queries)
        return {
            **self.vector_index.verify_exact(vectors, top_k),
            "query_token_statistics": self.encoder.token_statistics(queries),
            "scope": "untimed FAISS versus NumPy using identical encoded query vectors",
        }


def build_dense(
    documents: list[Document],
    config: DenseConfig,
    model_root: Path,
    cache_root: Path,
    *,
    allow_build: bool = True,
) -> DenseIndex:
    from retrieval_lab.encoders import SentenceEncoder

    started = perf_counter()
    encoder = SentenceEncoder(config, model_root)
    model_load_seconds = perf_counter() - started
    index = DenseIndex(documents, config, encoder, cache_root, allow_build=allow_build)
    index.preparation["model_load_seconds"] = model_load_seconds
    return index

"""A shared document-to-ranking boundary for comparable experiments."""

from pathlib import Path
from typing import Protocol

from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.hybrid import HybridConfig, build_hybrid
from retrieval_lab.reranking import RerankerConfig, build_reranked
from retrieval_lab.types import Document, SearchResult

RetrieverConfig = BM25Config | DenseConfig | HybridConfig | RerankerConfig


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]: ...


def load_config(path: Path, system: str) -> RetrieverConfig:
    if system == "bm25":
        return BM25Config.from_file(path)
    if system == "dense":
        return DenseConfig.from_file(path)
    if system == "hybrid":
        return HybridConfig.from_file(path)
    if system == "rerank":
        return RerankerConfig.from_file(path)
    raise ValueError(f"Unknown retrieval system: {system}")


def build_retriever(
    documents: list[Document],
    config: RetrieverConfig,
    model_root: Path = Path("models"),
    cache_root: Path = Path("indexes"),
    *,
    allow_build: bool = True,
) -> Retriever:
    if isinstance(config, BM25Config):
        return BM25Index(documents, config)
    if not allow_build:
        from retrieval_lab.cross_encoder import CrossEncoderScorer
        from retrieval_lab.dense import build_dense
        from retrieval_lab.hybrid import HybridIndex
        from retrieval_lab.reranking import RerankedIndex

        if isinstance(config, RerankerConfig):
            return RerankedIndex(
                build_retriever(
                    documents, config.retriever, model_root, cache_root, allow_build=False
                ),
                CrossEncoderScorer(config, model_root),
                config,
            )
        if isinstance(config, HybridConfig):
            return HybridIndex(
                BM25Index(documents, config.bm25),
                build_dense(documents, config.dense, model_root, cache_root, allow_build=False),
                config,
            )
        return build_dense(documents, config, model_root, cache_root, allow_build=False)
    if isinstance(config, HybridConfig):
        return build_hybrid(documents, config, model_root, cache_root)
    if isinstance(config, RerankerConfig):
        return build_reranked(documents, config, model_root, cache_root)
    from retrieval_lab.dense import build_dense

    return build_dense(documents, config, model_root, cache_root)

"""Bounded pretrained reranking. Candidate generation and labels remain separate."""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from retrieval_lab.bm25 import BM25Config
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.hybrid import HybridConfig, HybridIndex, _integer, build_hybrid
from retrieval_lab.types import Document, SearchResult


@dataclass(frozen=True)
class RerankerConfig:
    schema_version: int = 1
    system: str = "rerank"
    implementation_version: str = "1.0.0"
    model_id: str = "cross-encoder/ms-marco-TinyBERT-L2-v2"
    revision: str = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"
    rerank_depth: int = 50
    max_length: int = 512
    batch_size: int = 16
    device: str = "cpu"
    dtype: str = "float32"
    activation: str = "identity_raw_logit"
    truncation: str = "longest_first"
    torch_threads: int = 1
    seed: int = 42
    checkpoint_sha256: str | None = None
    retriever: HybridConfig = field(default_factory=HybridConfig)

    def __post_init__(self):
        for name, expected in {
            "schema_version": 1,
            "system": "rerank",
            "implementation_version": "1.0.0",
            "model_id": "cross-encoder/ms-marco-TinyBERT-L2-v2",
            "device": "cpu",
            "dtype": "float32",
            "activation": "identity_raw_logit",
            "truncation": "longest_first",
        }.items():
            if type(getattr(self, name)) is not type(expected) or getattr(self, name) != expected:
                raise ValueError(f"Unsupported reranker setting: {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Reranker revision must be an immutable 40-character commit")
        if self.checkpoint_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.checkpoint_sha256
        ):
            raise ValueError("Checkpoint identity must be a SHA-256 digest")
        for name in ["rerank_depth", "batch_size", "torch_threads"]:
            _integer(name, getattr(self, name), 1)
        _integer("max_length", self.max_length, 8)
        if self.max_length > 512:
            raise ValueError("TinyBERT supports at most 512 joint input tokens")
        if not isinstance(self.retriever, HybridConfig):
            raise ValueError("Reranking requires the declared HybridConfig candidate generator")
        if self.rerank_depth > self.retriever.candidate_depth:
            raise ValueError("rerank_depth must not exceed the hybrid candidate_depth")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.torch_threads != self.retriever.dense.torch_threads:
            raise ValueError("The reranker and dense encoder must share the CPU thread setting")

    @classmethod
    def from_file(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values: dict):
        values = dict(values)
        if "retriever" in values:
            child = dict(values["retriever"])
            for name, component in [("bm25", BM25Config), ("dense", DenseConfig)]:
                if name in child:
                    settings = dict(child[name])
                    settings["document_fields"] = tuple(
                        settings.get("document_fields", ("title", "text"))
                    )
                    child[name] = component(**settings)
            values["retriever"] = HybridConfig(**child)
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "retriever" and getattr(self, name) is not None
            },
            "retriever": self.retriever.to_dict(),
        }


class PairScorer(Protocol):
    fingerprint: dict

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...
    def token_statistics(self, pairs: list[tuple[str, str]]) -> dict: ...


def rerank_candidates(candidates: list[SearchResult], scores: list[float], top_k: int):
    _integer("top_k", top_k, 1)
    if len(scores) != len(candidates):
        raise ValueError("Reranker score count differs from candidate count")
    if len({hit.document.doc_id for hit in candidates}) != len(candidates):
        raise ValueError("Reranker candidates must have unique document IDs")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Reranker returned non-finite scores")
    order = sorted(
        range(len(candidates)), key=lambda i: (-scores[i], candidates[i].document.doc_id)
    )
    hits = [SearchResult(candidates[i].document, float(scores[i])) for i in order[:top_k]]
    trace = {
        "candidates": [
            {
                "doc_id": hit.document.doc_id,
                "retrieval_rank": rank,
                "retrieval_score": hit.score,
                "reranker_score": float(score),
            }
            for rank, (hit, score) in enumerate(zip(candidates, scores, strict=True), 1)
        ],
        "results": [
            {
                "doc_id": candidates[i].document.doc_id,
                "retrieval_rank": i + 1,
                "reranked_rank": rank,
                "raw_logit": float(scores[i]),
            }
            for rank, i in enumerate(order[:top_k], 1)
        ],
    }
    return hits, trace


class RerankedIndex:
    def __init__(self, retriever: HybridIndex, scorer: PairScorer, config: RerankerConfig):
        self.retriever, self.scorer, self.config = retriever, scorer, config
        self.preparation = {
            "retriever": retriever.preparation,
            "reranker": scorer.fingerprint,
            "rerank_depth": config.rerank_depth,
        }

    def search_with_trace(self, query: str, top_k: int = 10):
        _integer("top_k", top_k, 1)
        if top_k > self.config.rerank_depth:
            raise ValueError("top_k must not exceed rerank_depth")
        candidates, fusion_trace = self.retriever.search_with_trace(query, self.config.rerank_depth)
        pairs = [(query, f"{hit.document.title}\n{hit.document.text}") for hit in candidates]
        scores = self.scorer.score(pairs) if pairs else []
        hits, trace = rerank_candidates(candidates, scores, top_k)
        return hits, {
            **trace,
            "rerank_depth": self.config.rerank_depth,
            "fusion_trace": fusion_trace,
        }

    def search(self, query: str, top_k: int = 10):
        return self.search_with_trace(query, top_k)[0]

    def sanity_check(self, queries: list[str], top_k: int = 10):
        return self.retriever.sanity_check(queries, self.config.rerank_depth)


def build_reranked(
    documents: list[Document], config: RerankerConfig, model_root: Path, cache_root: Path
) -> RerankedIndex:
    from retrieval_lab.cross_encoder import CrossEncoderScorer

    return RerankedIndex(
        build_hybrid(documents, config.retriever, model_root, cache_root),
        CrossEncoderScorer(config, model_root),
        config,
    )

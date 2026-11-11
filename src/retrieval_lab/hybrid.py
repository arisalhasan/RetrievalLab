"""Label-free reciprocal rank fusion; existing component encoders stay unchanged."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.types import Document, SearchResult


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class HybridConfig:
    schema_version: int = 1
    system: str = "hybrid"
    implementation_version: str = "1.0.0"
    rank_constant: int = 60
    candidate_depth: int = 100
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)

    def __post_init__(self):
        for key, expected in [
            ("schema_version", 1),
            ("system", "hybrid"),
            ("implementation_version", "1.0.0"),
        ]:
            if type(getattr(self, key)) is not type(expected) or getattr(self, key) != expected:
                raise ValueError(f"Unsupported hybrid setting: {key}")
        _integer("rank_constant", self.rank_constant, 0)
        _integer("candidate_depth", self.candidate_depth, 1)
        if not isinstance(self.bm25, BM25Config) or not isinstance(self.dense, DenseConfig):
            raise ValueError("Hybrid components must be BM25Config and DenseConfig")
        defaults = BM25Config()
        for key in BM25Config.FIXED_FIELDS:
            if getattr(self.bm25, key) != getattr(defaults, key):
                raise ValueError(f"Unsupported BM25 setting: {key}")

    @classmethod
    def from_file(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, component in [("bm25", BM25Config), ("dense", DenseConfig)]:
            if name in values:
                child = dict(values[name])
                if "document_fields" in child:
                    child["document_fields"] = tuple(child["document_fields"])
                values[name] = component(**child)
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "system": self.system,
            "implementation_version": self.implementation_version,
            "rank_constant": self.rank_constant,
            "candidate_depth": self.candidate_depth,
            "bm25": self.bm25.to_dict(),
            "dense": self.dense.to_dict(),
        }


def fuse_rankings(
    rankings: dict[str, list[SearchResult]], rank_constant: int = 60, top_k: int = 10
) -> tuple[list[SearchResult], dict]:
    """Ranks are one-based. Missing documents contribute zero; raw scores are ignored.

    Duplicate IDs in one source and conflicting document payloads are component errors.
    The trace is JSON serializable and is built from this same retrieval, without labels.
    """
    _integer("rank_constant", rank_constant, 0)
    _integer("top_k", top_k, 1)
    documents, contributions = {}, defaultdict(dict)
    for source in sorted(rankings):
        seen = set()
        for rank, hit in enumerate(rankings[source], 1):
            did = hit.document.doc_id
            if did in seen:
                raise ValueError(f"Duplicate document ID in {source} ranking: {did}")
            if did in documents and documents[did] != hit.document:
                raise ValueError(f"Conflicting document content for ID: {did}")
            seen.add(did)
            documents[did] = hit.document
            contributions[did][source] = {
                "rank": rank,
                "contribution": 1 / (rank_constant + rank),
            }
    scores = {
        did: math.fsum(parts[source]["contribution"] for source in sorted(parts))
        for did, parts in contributions.items()
    }
    ordered = sorted(scores, key=lambda did: (-scores[did], did))[:top_k]
    hits = [SearchResult(documents[did], scores[did]) for did in ordered]
    trace = {
        "rank_constant": rank_constant,
        "source_candidate_ids": {
            name: [hit.document.doc_id for hit in rankings[name]] for name in sorted(rankings)
        },
        "union_size": len(documents),
        "results": [
            {"doc_id": did, "rrf_score": scores[did], "sources": contributions[did]}
            for did in ordered
        ],
    }
    return hits, trace


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]: ...


class HybridIndex:
    def __init__(self, bm25: Searcher, dense: Searcher, config: HybridConfig):
        self.bm25, self.dense, self.config = bm25, dense, config
        self.preparation = {
            "implementation": "serial BM25 + dense + equal-weight reciprocal rank fusion",
            "dense": getattr(dense, "preparation", {}),
            "rank_constant": config.rank_constant,
            "candidate_depth_per_source": config.candidate_depth,
        }

    def search_with_trace(self, query: str, top_k: int = 10) -> tuple[list[SearchResult], dict]:
        _integer("top_k", top_k, 1)
        if top_k > self.config.candidate_depth:
            raise ValueError("top_k must not exceed the configured candidate_depth")
        rankings = {
            "bm25": self.bm25.search(query, self.config.candidate_depth),
            "dense": self.dense.search(query, self.config.candidate_depth),
        }
        return fuse_rankings(rankings, self.config.rank_constant, top_k)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        return self.search_with_trace(query, top_k)[0]

    def sanity_check(self, queries: list[str], top_k: int = 10) -> dict:
        # Check the deeper dense candidate list used by fusion, not just final top 10.
        return {
            **self.dense.sanity_check(queries, self.config.candidate_depth),
            "component": "dense candidates used by hybrid",
            "checked_candidate_depth": self.config.candidate_depth,
        }


def build_hybrid(
    documents: list[Document], config: HybridConfig, model_root: Path, cache_root: Path
) -> HybridIndex:
    from retrieval_lab.dense import build_dense

    return HybridIndex(
        BM25Index(documents, config.bm25),
        build_dense(documents, config.dense, model_root, cache_root),
        config,
    )

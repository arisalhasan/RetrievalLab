"""An inverted-index BM25 baseline with positive IDF and stable tie breaking."""

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from retrieval_lab.types import Document, SearchResult


def tokenize(text: str) -> list[str]:
    """Unicode letter/digit words; no stemming or stopword removal."""
    return re.findall(r"[^\W_]+", text.casefold())


@dataclass(frozen=True)
class BM25Config:
    schema_version: int = 1
    system: str = "bm25"
    implementation_version: str = "1.0.0"
    tokenizer: str = "unicode-word-casefold-v1"
    document_fields: tuple[str, ...] = ("title", "text")
    idf: str = "log1p((N-df+0.5)/(df+0.5))"
    query_term_weighting: str = "binary"
    k1: float = 1.2
    b: float = 0.75
    seed: int = 42  # Reserved for later stochastic models; BM25 uses no randomness.

    FIXED_FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "system",
        "implementation_version",
        "tokenizer",
        "document_fields",
        "idf",
        "query_term_weighting",
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and between zero and one")

    @classmethod
    def from_file(cls, path: str | Path) -> "BM25Config":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if "document_fields" in values:
            values["document_fields"] = tuple(values["document_fields"])
        config = cls(**values)
        defaults = cls()
        for name in cls.FIXED_FIELDS:
            if getattr(config, name) != getattr(defaults, name):
                raise ValueError(f"Unsupported BM25 setting: {name}")
        return config

    def to_dict(self) -> dict:
        return {**asdict(self), "document_fields": list(self.document_fields)}


class BM25Index:
    """Fit on documents only. Labels never enter indexing or scoring."""

    def __init__(self, documents: list[Document], config: BM25Config | None = None):
        if not documents:
            raise ValueError("Cannot index an empty corpus")
        self.config = config or BM25Config()
        self.documents = {doc.doc_id: doc for doc in sorted(documents, key=lambda d: d.doc_id)}
        if len(self.documents) != len(documents):
            raise ValueError("Document IDs must be unique")
        postings = defaultdict(list)
        lengths = {}
        for doc in self.documents.values():
            terms = tokenize(f"{doc.title}\n{doc.text}")
            lengths[doc.doc_id] = len(terms)
            for term, count in Counter(terms).items():
                postings[term].append((doc.doc_id, count))
        self.average_length = sum(lengths.values()) / len(lengths)
        if self.average_length == 0:
            raise ValueError("Corpus contains no searchable words")
        self._lengths = lengths
        self._postings = dict(postings)
        self._idf = {
            term: math.log1p((len(documents) - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores: dict[str, float] = defaultdict(float)
        k1, b = self.config.k1, self.config.b
        # Sorting also makes floating-point accumulation repeatable across processes.
        for term in sorted(set(tokenize(query))):
            for doc_id, frequency in self._postings.get(term, ()):
                norm = 1 - b + b * self._lengths[doc_id] / self.average_length
                scores[doc_id] += self._idf[term] * frequency * (k1 + 1) / (frequency + k1 * norm)
        # Return matches only; never pad unknown queries with arbitrary zero-score docs.
        ranked = heapq.nsmallest(top_k, scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [SearchResult(self.documents[doc_id], scores[doc_id]) for doc_id in ranked]

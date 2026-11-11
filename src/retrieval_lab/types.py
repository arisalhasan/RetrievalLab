"""Shared value objects; retrieval does not depend on relevance judgments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float

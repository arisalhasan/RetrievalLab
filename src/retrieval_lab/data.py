"""Read the official BEIR JSONL/TSV format with explicit split selection."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from retrieval_lab.types import Document

OFFICIAL_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")


class DatasetError(ValueError):
    """The dataset is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Dataset:
    corpus: dict[str, Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    split: str
    file_sha256: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON at {path.name}:{line_no}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("_id"), str) or not row["_id"]:
                raise DatasetError(f"Missing string _id at {path.name}:{line_no}")
            if row["_id"] in records:
                raise DatasetError(f"Duplicate ID {row['_id']} in {path.name}")
            records[row["_id"]] = row
    if not records:
        raise DatasetError(f"Empty file: {path.name}")
    return records


def load_dataset(folder: str | Path, split: str = "train") -> Dataset:
    """Load only the selected split's judgments; never merge train and test.

    All documents are searchable. Queries are filtered by the chosen qrels,
    matching BEIR's GenericDataLoader behavior. No test qrels are read in train mode.
    """
    if split not in {"train", "test"}:
        raise DatasetError("SciFact has official train/test splits; no official dev split.")
    folder = Path(folder)
    files = ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv")
    for name in files:
        if not (folder / name).is_file():
            raise DatasetError(f"Missing {folder / name}. Run: retrievallab download")
    try:
        corpus = {}
        for doc_id, row in _read_jsonl(folder / files[0]).items():
            title, text = row.get("title", ""), row.get("text")
            if (
                not isinstance(title, str)
                or not isinstance(text, str)
                or not (title + text).strip()
            ):
                raise DatasetError(f"Document {doc_id} needs string title/text and some content")
            corpus[doc_id] = Document(doc_id, title, text)
        all_queries = _read_jsonl(folder / files[1])
        for query_id, row in all_queries.items():
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise DatasetError(f"Query {query_id} needs nonempty text")
        qrels: dict[str, dict[str, int]] = {}
        with (folder / files[2]).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader, None) != ["query-id", "corpus-id", "score"]:
                raise DatasetError(
                    "Qrels header must be: query-id, corpus-id, score (tab-separated)"
                )
            for line_no, row in enumerate(reader, 2):
                if len(row) != 3:
                    raise DatasetError(f"Invalid qrels row at line {line_no}")
                query_id, doc_id, raw_score = row
                try:
                    score = int(raw_score)
                except ValueError as exc:
                    raise DatasetError(f"Invalid relevance score at line {line_no}") from exc
                if query_id not in all_queries or doc_id not in corpus or score < 0:
                    raise DatasetError(
                        f"Unknown query/document or negative score at line {line_no}"
                    )
                judgments = qrels.setdefault(query_id, {})
                if doc_id in judgments:
                    raise DatasetError(f"Duplicate judgment at line {line_no}")
                judgments[doc_id] = score
        if not qrels:
            raise DatasetError(f"No judgments for split {split}")
        queries = {qid: all_queries[qid]["text"] for qid in sorted(qrels)}
        return Dataset(corpus, queries, qrels, split, {f: sha256_file(folder / f) for f in files})
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read dataset at {folder}: {exc}") from exc

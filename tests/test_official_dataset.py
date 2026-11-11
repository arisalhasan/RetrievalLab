"""Opt-in real-data smoke checks. No test relevance labels are opened."""

import os
from pathlib import Path

import pytest

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.data import load_dataset
from retrieval_lab.download import verify_cached_dataset
from retrieval_lab.splits import load_split


@pytest.mark.dataset
@pytest.mark.skipif(os.environ.get("RUN_SCIFACT_TESTS") != "1", reason="Set RUN_SCIFACT_TESTS=1")
def test_official_scifact_train_search():
    folder = Path(os.environ.get("RETRIEVALLAB_DATA_DIR", "data/scifact"))
    verify_cached_dataset(folder)
    dataset = load_dataset(folder, "train")
    assert len(dataset.corpus) == 5183
    assert len(dataset.queries) == 809
    index = BM25Index(list(dataset.corpus.values()))
    query_id = sorted(dataset.queries)[0]
    hits = index.search(dataset.queries[query_id], 10)
    assert 0 < len(hits) <= 10
    assert all(hit.document.doc_id in dataset.corpus for hit in hits)
    assert hits == index.search(dataset.queries[query_id], 10)


@pytest.mark.dataset
@pytest.mark.skipif(os.environ.get("RUN_SCIFACT_TESTS") != "1", reason="Set RUN_SCIFACT_TESTS=1")
def test_saved_official_training_partition_has_no_evidence_overlap():
    folder = Path(os.environ.get("RETRIEVALLAB_DATA_DIR", "data/scifact"))
    dataset = load_dataset(folder, "train")
    root = Path(__file__).resolve().parents[1]
    split = load_split(root / "configs/splits/scifact-train-v1.json", dataset)
    assert split["counts"]["train"] == 647
    assert split["counts"]["validation"] == 162
    train_docs = {
        did
        for qid in split["train_query_ids"]
        for did, grade in dataset.qrels[qid].items()
        if grade > 0
    }
    validation_docs = {
        did
        for qid in split["validation_query_ids"]
        for did, grade in dataset.qrels[qid].items()
        if grade > 0
    }
    assert not train_docs & validation_docs

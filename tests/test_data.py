import json

import pytest

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.data import DatasetError, load_dataset


def test_load_and_search_integration(mini_dataset):
    dataset = load_dataset(mini_dataset)
    index = BM25Index(list(dataset.corpus.values()))
    assert len(dataset.corpus) == 3
    assert dataset.split == "train"
    assert dataset.queries == {"train-1": "cell mitosis"}
    assert dataset.qrels == {"train-1": {"d1": 1}}
    assert "qrels/test.tsv" not in dataset.file_sha256
    assert len(dataset.file_sha256["corpus.jsonl"]) == 64
    assert {r.document.doc_id for r in index.search(dataset.queries["train-1"])} == {"d1", "d3"}


def test_train_loader_never_reads_test_labels(mini_dataset):
    (mini_dataset / "qrels/test.tsv").write_text(
        "intentionally unreadable labels", encoding="utf-8"
    )
    assert list(load_dataset(mini_dataset).queries) == ["train-1"]


def test_test_loading_is_explicit_and_separate(mini_dataset):
    assert list(load_dataset(mini_dataset, "test").queries) == ["test-1"]
    with pytest.raises(DatasetError, match="no official dev"):
        load_dataset(mini_dataset, "dev")


def test_missing_file_has_recovery_command(tmp_path):
    with pytest.raises(DatasetError, match="retrievallab download"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "row",
    [
        "missing\td1\t1",
        "train-1\tmissing\t1",
        "train-1\td1\t-1",
        "train-1\td1\tbad",
        "train-1\td1",
        "train-1\td1\t1\ntrain-1\td1\t1",
    ],
)
def test_invalid_judgments_are_rejected(mini_dataset, row):
    (mini_dataset / "qrels/train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + row + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError):
        load_dataset(mini_dataset)


def test_duplicate_document_ids_rejected(mini_dataset):
    path = mini_dataset / "corpus.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"_id": "d1", "title": "x", "text": "y"}) + "\n")
    with pytest.raises(DatasetError, match="Duplicate ID"):
        load_dataset(mini_dataset)


def test_invalid_json_has_location(mini_dataset):
    (mini_dataset / "corpus.jsonl").write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="corpus.jsonl:1"):
        load_dataset(mini_dataset)

from dataclasses import replace

import pytest

from retrieval_lab.data import Dataset, DatasetError
from retrieval_lab.splits import load_split, make_split, write_split
from retrieval_lab.types import Document


@pytest.fixture
def development():
    corpus = {
        key: Document(key, key, text)
        for key, text in {
            "d1": "cell division mitosis",
            "d2": "stars planets orbit",
            "d3": "proteins fold",
            "d4": "bone density calcium",
        }.items()
    }
    queries = {
        "a": "Cells divide by mitosis",
        "b": "The process of cell division",
        "c": "Planets orbit stars",
        "d": "Proteins fold into complex shapes",
        "e": "proteins fold into complex shapes!",
        "f": "Calcium and bone density",
    }
    qrels = {
        qid: {did: 1}
        for qid, did in zip(queries, ["d1", "d1", "d2", "d3", "d4", "d4"], strict=True)
    }
    return Dataset(corpus, queries, qrels, "train", {"synthetic": "fixture-v1"})


def test_groups_prevent_evidence_and_near_duplicate_leakage(development):
    split = make_split(development)
    train, val = set(split["train_query_ids"]), set(split["validation_query_ids"])
    assert train and val and not train & val
    assert train | val == set(development.queries)
    assert ("a" in train) == ("b" in train)  # Shared positive document.
    assert ("d" in train) == ("e" in train)  # Same words, different evidence.
    assert ("e" in train) == ("f" in train)  # Transitive shared-evidence link.
    train_docs = {doc for qid in train for doc in development.qrels[qid]}
    val_docs = {doc for qid in val for doc in development.qrels[qid]}
    assert not train_docs & val_docs


def test_split_is_independent_of_input_order_and_can_be_verified(development, tmp_path):
    split = make_split(development)
    reordered = replace(development, queries=dict(reversed(list(development.queries.items()))))
    assert make_split(reordered) == split
    path = tmp_path / "split.json"
    write_split(split, path)
    assert load_split(path, development) == split
    with pytest.raises(FileExistsError):
        write_split(split, path)


def test_tampered_split_and_changed_inputs_are_rejected(development, tmp_path):
    path = tmp_path / "split.json"
    split = make_split(development)
    write_split(split, path)
    changed = replace(development, file_sha256={"synthetic": "changed"})
    with pytest.raises(DatasetError, match="does not match"):
        load_split(path, changed)
    split["validation_query_ids"].append("not-a-source-query")
    other = tmp_path / "bad.json"
    write_split(split, other)
    with pytest.raises(DatasetError, match="does not match"):
        load_split(other, development)


def test_test_split_and_single_connected_group_are_rejected(development):
    with pytest.raises(DatasetError, match="official train"):
        make_split(replace(development, split="test"))
    same = {qid: {"d1": 1} for qid in development.queries}
    with pytest.raises(DatasetError, match="two independent"):
        make_split(replace(development, qrels=same))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"near_duplicate_threshold": 0},
        {"near_duplicate_threshold": float("nan")},
        {"seed": True},
    ],
)
def test_invalid_split_settings(development, kwargs):
    with pytest.raises(ValueError):
        make_split(development, **kwargs)

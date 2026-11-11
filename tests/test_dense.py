import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab.data import DatasetError, sha256_file
from retrieval_lab.dense import DenseIndex, prepare_embeddings
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.types import Document


class SyntheticEncoder:
    """External model boundary only: synthetic vectors, never benchmark results."""

    def __init__(self):
        self.fingerprint = {"backend": "synthetic-unit-test-vectors", "version": "1"}
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        values = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            position = 0 if "planet" in text.lower() else 1
            values[row, position] = 1
        return values

    def token_statistics(self, texts):
        return {"synthetic": True, "texts": len(texts), "texts_truncated": 0}


@pytest.fixture
def documents():
    return [Document("b", "Cells", "cell division"), Document("a", "Planets", "planets orbit")]


def test_embedding_cache_reuses_verified_rows_without_running_encoder(documents, tmp_path):
    first_encoder = SyntheticEncoder()
    vectors, first = prepare_embeddings(documents, DenseConfig(), first_encoder, tmp_path)
    second_encoder = SyntheticEncoder()
    cached, second = prepare_embeddings(documents[::-1], DenseConfig(), second_encoder, tmp_path)
    assert not first["cache_hit"] and second["cache_hit"]
    assert second_encoder.calls == []
    assert np.array_equal(vectors, cached)
    folder = tmp_path / first["cache_key"]
    assert json.loads((folder / "doc_ids.json").read_text()) == ["a", "b"]


def test_cache_changes_with_corpus_config_and_model_fingerprint(documents, tmp_path):
    encoder = SyntheticEncoder()
    _, first = prepare_embeddings(documents, DenseConfig(), encoder, tmp_path)
    _, config_changed = prepare_embeddings(
        documents, DenseConfig(max_seq_length=128), encoder, tmp_path
    )
    modified = [Document("b", "Changed title", "cell division"), documents[1]]
    _, corpus_changed = prepare_embeddings(modified, DenseConfig(), encoder, tmp_path)
    encoder.fingerprint["version"] = "2"
    _, model_changed = prepare_embeddings(documents, DenseConfig(), encoder, tmp_path)
    assert (
        len({row["cache_key"] for row in [first, config_changed, corpus_changed, model_changed]})
        == 4
    )


def test_corrupted_cache_is_refused(documents, tmp_path):
    _, info = prepare_embeddings(documents, DenseConfig(), SyntheticEncoder(), tmp_path)
    folder = tmp_path / info["cache_key"]
    (folder / "embeddings.npy").write_bytes(b"broken")
    with pytest.raises(DatasetError, match="checksum mismatch"):
        prepare_embeddings(documents, DenseConfig(), SyntheticEncoder(), tmp_path)


def test_cache_checks_row_alignment_even_if_file_hash_was_updated(documents, tmp_path):
    _, info = prepare_embeddings(documents, DenseConfig(), SyntheticEncoder(), tmp_path)
    folder = tmp_path / info["cache_key"]
    path = folder / "doc_ids.json"
    path.write_text('["b", "a"]', encoding="utf-8")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["file_sha256"]["doc_ids.json"] = sha256_file(path)
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetError, match="canonical document IDs"):
        prepare_embeddings(documents, DenseConfig(), SyntheticEncoder(), tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="Install dense dependencies")
def test_dense_search_encodes_each_query_and_handles_empty_input(documents, tmp_path):
    encoder = SyntheticEncoder()
    index = DenseIndex(documents, DenseConfig(), encoder, tmp_path)
    first = index.search("planets", 1)
    second = index.search("planets", 1)
    assert first == second and first[0].document.doc_id == "a"
    assert [len(call) for call in encoder.calls] == [2, 1, 1]
    assert index.search(" !!! ") == []
    assert index.sanity_check(["planets", "cells"])["passed"]


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("RUN_DENSE_MODEL_TESTS") != "1", reason="Opt-in local model check"
)
def test_real_pinned_encoder_end_to_end(documents, tmp_path):
    from retrieval_lab.encoders import SentenceEncoder

    root = Path(__file__).resolve().parents[1]
    encoder = SentenceEncoder(DenseConfig(), root / "models")
    vectors = encoder.encode(["planets orbit stars", "cells divide"])
    assert vectors.shape == (2, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    index = DenseIndex(documents, DenseConfig(), encoder, tmp_path)
    assert index.search("planets orbit stars", 1)[0].document.doc_id == "a"
    assert index.sanity_check(["planets orbit stars", "cells divide"])["passed"]
    reused = DenseIndex(documents, DenseConfig(), encoder, tmp_path, allow_build=False)
    assert reused.preparation["cache_hit"]

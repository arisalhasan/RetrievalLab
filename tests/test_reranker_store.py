import hashlib
import io
import json

import pytest

from retrieval_lab import reranker_store as store
from retrieval_lab.data import DatasetError
from retrieval_lab.reranking import RerankerConfig


def assets(monkeypatch):
    files = {name: b"synthetic test asset" for name in store.MODEL_FILES}
    files["config.json"] = json.dumps(
        {
            "architectures": ["BertForSequenceClassification"],
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "id2label": {"0": "LABEL_0"},
        }
    ).encode()
    rows = []
    for name, value in files.items():
        row = {
            "rfilename": name,
            "size": len(value),
            "blobId": hashlib.sha1(
                f"blob {len(value)}\0".encode() + value, usedforsecurity=False
            ).hexdigest(),
        }
        if name == "model.safetensors":
            row["lfs"] = {"sha256": hashlib.sha256(value).hexdigest()}
            monkeypatch.setattr(store, "WEIGHT_SHA256", row["lfs"]["sha256"])
        rows.append(row)
    payload = json.dumps({"sha": RerankerConfig().revision, "siblings": rows}).encode()
    monkeypatch.setattr(
        store.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    monkeypatch.setattr(
        store,
        "_fetch_file",
        lambda url, path, size, context: path.write_bytes(files[url.rsplit("/", 1)[1]]),
    )
    return files


def test_checked_download_cache_and_tamper_rejection(tmp_path, monkeypatch):
    assets(monkeypatch)
    config = RerankerConfig()
    folder = store.download_model(config, tmp_path)
    assert store.download_model(config, tmp_path) == folder
    assert store.verify_model(config, tmp_path)["revision"] == config.revision
    (folder / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(DatasetError, match="checksum"):
        store.verify_model(config, tmp_path)


def test_incomplete_download_never_installs(tmp_path, monkeypatch):
    files = assets(monkeypatch)
    files["model.safetensors"] = b"broken"
    with pytest.raises(DatasetError, match="Incomplete"):
        store.download_model(RerankerConfig(), tmp_path)
    assert not store.model_folder(RerankerConfig(), tmp_path).exists()

import hashlib
import io
import json

import pytest

from retrieval_lab import model_store
from retrieval_lab.data import DatasetError
from retrieval_lab.dense_config import DenseConfig


def synthetic_model(monkeypatch):
    files = {name: b"synthetic unit-test asset" for name in model_store.MODEL_FILES}
    files["modules.json"] = json.dumps(
        [
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling"},
            {"type": "sentence_transformers.models.Normalize"},
        ]
    ).encode()
    files["1_Pooling/config.json"] = (
        b'{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}'
    )
    rows = []
    for name, content in files.items():
        blob = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content, usedforsecurity=False
        ).hexdigest()
        row = {"rfilename": name, "size": len(content), "blobId": blob}
        if name == "model.safetensors":
            row["lfs"] = {"sha256": hashlib.sha256(content).hexdigest()}
            monkeypatch.setattr(model_store, "WEIGHT_SHA256", row["lfs"]["sha256"])
        rows.append(row)
    payload = json.dumps({"sha": DenseConfig().revision, "siblings": rows}).encode()

    def response(url, **kwargs):
        url = url.full_url if hasattr(url, "full_url") else url
        if "/api/models/" in url:
            return io.BytesIO(payload)
        name = url.split(f"/resolve/{DenseConfig().revision}/")[1]
        return io.BytesIO(files[name])

    monkeypatch.setattr(model_store.urllib.request, "urlopen", response)

    def fetch(url, path, expected_size, context):
        name = url.split(f"/resolve/{DenseConfig().revision}/")[1]
        path.write_bytes(files[name])

    monkeypatch.setattr(model_store, "_fetch_file", fetch)
    return files


def test_model_download_checks_upstream_hashes_and_cached_files(tmp_path, monkeypatch):
    synthetic_model(monkeypatch)
    config = DenseConfig()
    path = model_store.download_model(config, tmp_path)
    assert model_store.verify_model(config, tmp_path)["revision"] == config.revision
    assert model_store.download_model(config, tmp_path) == path
    (path / "tokenizer.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(DatasetError, match="checksum mismatch"):
        model_store.verify_model(config, tmp_path)


def test_model_download_does_not_install_corrupted_files(tmp_path, monkeypatch):
    files = synthetic_model(monkeypatch)
    # The upstream metadata stays fixed while the response is changed.
    files["model.safetensors"] = b"wrong bytes"
    with pytest.raises(DatasetError, match="Incomplete model file"):
        model_store.download_model(DenseConfig(), tmp_path)
    assert not model_store.model_folder(DenseConfig(), tmp_path).exists()


@pytest.mark.parametrize("honor_range", [True, False])
def test_weights_transport_checks_each_bounded_range(tmp_path, monkeypatch, honor_range):
    payload = b"x" * (8 * 1024**2 + 17)
    requested = []

    def response(request, **kwargs):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        requested.append((start, end))
        stream = io.BytesIO(payload[start : end + 1])
        stream.status = 206 if honor_range else 200
        stream.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        return stream

    monkeypatch.setattr(model_store.urllib.request, "urlopen", response)
    path = tmp_path / "weights"
    if not honor_range:
        with pytest.raises(DatasetError, match="byte range"):
            model_store._fetch_file("https://example.test/weights", path, len(payload), None)
    else:
        model_store._fetch_file("https://example.test/weights", path, len(payload), None)
        assert path.read_bytes() == payload
        assert requested == [(0, 4194303), (4194304, 8388607), (8388608, 8388624)]

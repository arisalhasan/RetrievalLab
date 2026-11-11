import hashlib
import io
import json
import zipfile

import pytest

from retrieval_lab import download
from retrieval_lab.data import OFFICIAL_FILES, DatasetError


def make_archive(folder, path):
    with zipfile.ZipFile(path, "w") as bundle:
        for name in OFFICIAL_FILES:
            bundle.write(folder / name, f"scifact/{name}")
        bundle.writestr("../../escape.txt", "must never be extracted")
    return path


def test_verified_install_is_cached_and_only_extracts_known_files(
    mini_dataset, tmp_path, monkeypatch
):
    archive = make_archive(mini_dataset, tmp_path / "synthetic.zip")
    md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    monkeypatch.setattr(download, "SCIFACT_MD5", md5)  # Synthetic fixture only.
    root = tmp_path / "cache"
    installed = download.download_scifact(root, archive)
    assert (installed / "corpus.jsonl").is_file()
    assert not (tmp_path / "escape.txt").exists()
    manifest = json.loads((installed / "manifest.json").read_text())
    assert manifest["document_count"] == 3
    assert manifest["train_query_count"] == 1
    assert download.download_scifact(root) == installed  # Must not use the network.
    (installed / "corpus.jsonl").write_text("corrupted", encoding="utf-8")
    with pytest.raises(DatasetError, match="checksum mismatch"):
        download.download_scifact(root)


def test_checksum_failure_leaves_no_installed_or_temporary_data(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"bad archive")
    root = tmp_path / "cache"
    with pytest.raises(DatasetError, match="checksum mismatch"):
        download.download_scifact(root, archive)
    assert list(root.iterdir()) == []


def test_network_download_boundary(mini_dataset, tmp_path, monkeypatch):
    archive = make_archive(mini_dataset, tmp_path / "synthetic.zip")
    content = archive.read_bytes()
    monkeypatch.setattr(
        download, "SCIFACT_MD5", hashlib.md5(content, usedforsecurity=False).hexdigest()
    )
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    assert download.download_scifact(tmp_path / "cache").is_dir()

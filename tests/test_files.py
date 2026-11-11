from pathlib import Path

import pytest

from retrieval_lab.files import install_directory


def test_staged_install_retries_a_windows_sharing_lock(tmp_path, monkeypatch):
    source = tmp_path / "stage" / "artifact"
    source.mkdir(parents=True)
    (source / "evidence.txt").write_text("keep", encoding="utf-8")
    rename = Path.rename
    calls = []

    def transient(path, destination):
        calls.append(destination)
        if len(calls) == 1:
            exc = PermissionError("Transient Windows sharing lock")
            exc.winerror = 32
            raise exc
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", transient)
    install_directory(source, tmp_path / "final", tmp_path)
    assert (tmp_path / "final/evidence.txt").read_text() == "keep"
    assert len(calls) == 2


def test_staged_install_refuses_existing_or_out_of_root_destinations(tmp_path):
    source = tmp_path / "stage"
    source.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        install_directory(source, existing, tmp_path)
    with pytest.raises(ValueError):
        install_directory(source, tmp_path.parent / "outside", tmp_path)

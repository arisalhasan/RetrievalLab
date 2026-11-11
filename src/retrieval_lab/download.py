"""Download the published SciFact archive once and record its provenance."""

import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from retrieval_lab.data import OFFICIAL_FILES, DatasetError, load_dataset, sha256_file
from retrieval_lab.files import install_directory

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# Published in the BEIR dataset table. SHA-256 is also recorded after download.
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_cached_dataset(folder: Path) -> dict:
    """Refuse a partial or edited cache instead of silently reusing it."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["archive_md5"] != SCIFACT_MD5 or manifest["source_url"] != SCIFACT_URL:
            raise DatasetError("Cache does not identify the published SciFact archive")
        for name in OFFICIAL_FILES:
            if sha256_file(folder / name) != manifest["file_sha256"][name]:
                raise DatasetError(f"Cache checksum mismatch: {name}")
        return manifest
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DatasetError(
            f"Cannot verify {folder}: {exc}. Use a new --data-root or restore the original cache."
        ) from exc


def download_scifact(data_root: str | Path = "data", archive: Path | None = None) -> Path:
    """Install into data_root/scifact atomically; existing files are never overwritten.

    A local archive is useful when the machine is offline. It must pass the same
    published checksum as an online download. Only four known data files are copied.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "scifact"
    if target.exists():
        verify_cached_dataset(target)
        return target
    try:
        with tempfile.TemporaryDirectory(prefix=".scifact-", dir=root) as temp:
            stage = Path(temp)
            zip_path = archive if archive is not None else stage / "scifact.zip"
            if archive is None:
                request = urllib.request.Request(
                    SCIFACT_URL, headers={"User-Agent": "RetrievalLab/0.1"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    zip_path.open("wb") as out,
                ):
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > MAX_ARCHIVE_BYTES:
                            raise DatasetError("SciFact archive exceeded the 50 MiB download limit")
                        out.write(block)
            if _md5_file(zip_path) != SCIFACT_MD5:
                raise DatasetError("SciFact archive checksum mismatch; nothing was installed")
            unpacked = stage / "scifact"
            unpacked.mkdir()
            with zipfile.ZipFile(zip_path) as bundle:
                for name in OFFICIAL_FILES:
                    # No extractall: archive paths can never choose destinations.
                    member = bundle.getinfo(f"scifact/{name}")
                    if member.file_size > MAX_ARCHIVE_BYTES:
                        raise DatasetError(f"Unexpectedly large archive member: {name}")
                    destination = unpacked / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open("wb") as out:
                        shutil.copyfileobj(source, out)
            # Validate the development split only. Hash test bytes without inspecting labels.
            development = load_dataset(unpacked, split="train")
            manifest = {
                "schema_version": 1,
                "dataset": "BEIR/SciFact",
                "source_url": SCIFACT_URL,
                "archive_md5": SCIFACT_MD5,
                "archive_sha256": sha256_file(zip_path),
                "downloaded_at_utc": datetime.now(UTC).isoformat(),
                "official_splits": ["train", "test"],
                "document_count": len(development.corpus),
                "train_query_count": len(development.queries),
                "file_sha256": {name: sha256_file(unpacked / name) for name in OFFICIAL_FILES},
            }
            (unpacked / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            install_directory(unpacked, target, root)
        return target
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, KeyError) as exc:
        raise DatasetError(f"Could not install SciFact: {exc}") from exc

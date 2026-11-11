"""Explicit public model downloads, pinned by commit and upstream blob hashes."""

import hashlib
import json
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from retrieval_lab.data import DatasetError, sha256_file
from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.files import install_directory

MODEL_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
    "README.md",
    "data_config.json",
)
WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def model_folder(config: DenseConfig, root: Path) -> Path:
    return root / f"all-MiniLM-L6-v2-{config.revision}"


def _verify_folder(config: DenseConfig, folder: Path) -> dict:
    try:
        manifest = json.loads((folder / "retrievallab-model.json").read_text(encoding="utf-8"))
        if manifest["model_id"] != config.model_id or manifest["revision"] != config.revision:
            raise DatasetError("Model identity does not match the selected config")
        if set(manifest["file_sha256"]) != set(MODEL_FILES):
            raise DatasetError("Model manifest does not list exactly the expected files")
        for name in MODEL_FILES:
            if sha256_file(folder / name) != manifest["file_sha256"][name]:
                raise DatasetError(f"Model file checksum mismatch: {name}")
        if config.revision == DenseConfig().revision:
            if manifest["file_sha256"]["model.safetensors"] != WEIGHT_SHA256:
                raise DatasetError("Model weights differ from the declared upstream artifact")
        modules = json.loads((folder / "modules.json").read_text(encoding="utf-8"))
        expected = [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        ]
        if [module["type"] for module in modules] != expected:
            raise DatasetError("Unexpected Sentence Transformers module architecture")
        pooling = json.loads((folder / "1_Pooling/config.json").read_text(encoding="utf-8"))
        if not pooling.get("pooling_mode_mean_tokens") or pooling.get("pooling_mode_cls_token"):
            raise DatasetError("Expected mean-token pooling")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(
            f"Verified local model unavailable. Run: retrievallab model-download. Details: {exc}"
        ) from exc


def verify_model(config: DenseConfig, root: Path) -> dict:
    return _verify_folder(config, model_folder(config, root))


def _fetch_file(url: str, path: Path, expected_size: int, context: ssl.SSLContext) -> None:
    # A socket timeout allows slow transfers without imposing a short total deadline.
    # Bounded ranges avoid CDN stalls observed for the single large weights response.
    if expected_size > 8 * 1024**2:
        with path.open("wb") as out:
            for start in range(0, expected_size, 4 * 1024**2):
                end = min(start + 4 * 1024**2, expected_size) - 1
                request = urllib.request.Request(
                    f"{url}?download=true&retrievallab_part={start}",
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "RetrievalLab/0.3"},
                )
                with urllib.request.urlopen(request, context=context, timeout=120) as response:
                    expected_range = f"bytes {start}-{end}/{expected_size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise DatasetError("Model server did not honor the requested byte range")
                    chunk = response.read(end - start + 2)
                if len(chunk) != end - start + 1:
                    raise DatasetError("Incomplete model download range")
                out.write(chunk)
                print(
                    f"Model weights: {end + 1:,}/{expected_size:,} bytes",
                    file=sys.stderr,
                    flush=True,
                )
        return
    request = urllib.request.Request(url, headers={"User-Agent": "RetrievalLab/0.3"})
    size = 0
    with (
        urllib.request.urlopen(request, context=context, timeout=120) as response,
        path.open("wb") as out,
    ):
        while chunk := response.read(65536):
            size += len(chunk)
            if size > expected_size:
                raise DatasetError("Model response exceeded the upstream file size")
            out.write(chunk)


def download_model(config: DenseConfig, root: Path = Path("models")) -> Path:
    """No account/token and no remote Python code. Download only known assets.

    TLS remains verified. Model loading later uses only these local files.
    """
    target = model_folder(config, root)
    if target.exists():
        verify_model(config, root)
        return target
    root.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    api = (
        f"https://huggingface.co/api/models/{config.model_id}/revision/{config.revision}?blobs=true"
    )
    try:
        with urllib.request.urlopen(api, context=context, timeout=60) as response:
            upstream = json.load(response)
        if upstream["sha"] != config.revision:
            raise DatasetError("Hub returned a different model revision")
        blobs = {row["rfilename"]: row for row in upstream["siblings"]}
        with tempfile.TemporaryDirectory(prefix=".model-", dir=root) as temp:
            stage = Path(temp) / "model"
            stage.mkdir()
            hashes = {}
            for name in MODEL_FILES:
                metadata = blobs[name]
                if metadata["size"] > 120 * 1024**2:
                    raise DatasetError(f"Unexpectedly large model file: {name}")
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/{config.model_id}/resolve/{config.revision}/{name}"
                _fetch_file(url, path, metadata["size"], context)
                size = path.stat().st_size
                if size != metadata["size"]:
                    raise DatasetError(f"Incomplete model file: {name}")
                hashes[name] = sha256_file(path)
                if metadata.get("lfs"):
                    if hashes[name] != metadata["lfs"]["sha256"]:
                        raise DatasetError(f"Upstream SHA-256 mismatch: {name}")
                else:
                    blob = hashlib.sha1(usedforsecurity=False)
                    blob.update(f"blob {size}\0".encode())
                    blob.update(path.read_bytes())
                    if blob.hexdigest() != metadata["blobId"]:
                        raise DatasetError(f"Upstream Git-blob mismatch: {name}")
            manifest = {
                "schema_version": 1,
                "model_id": config.model_id,
                "revision": config.revision,
                "file_sha256": hashes,
                "downloaded_at_utc": datetime.now(UTC).isoformat(),
                "source_api": api,
                "weights_format": "safetensors",
            }
            (stage / "retrievallab-model.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            _verify_folder(config, stage)
            install_directory(stage, target, root)
        verify_model(config, root)
        return target
    except (OSError, KeyError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise DatasetError(f"Could not download the pinned model: {exc}") from exc

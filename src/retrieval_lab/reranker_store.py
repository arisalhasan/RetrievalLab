"""Pinned, checked TinyBERT assets. Reuses the existing bounded TLS transport."""

import hashlib
import json
import ssl
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from retrieval_lab.data import DatasetError, sha256_file
from retrieval_lab.files import install_directory
from retrieval_lab.model_store import _fetch_file
from retrieval_lab.reranking import RerankerConfig

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "README.md",
)
WEIGHT_SHA256 = "a0e7364ddf91ff7028f1102e1b91ac7a72e3db4061241bd84efe45c72c9af03a"


def model_folder(config: RerankerConfig, root: Path) -> Path:
    if config.checkpoint_sha256:
        return root / "checkpoints" / config.checkpoint_sha256
    return root / f"ms-marco-TinyBERT-L2-v2-{config.revision}"


def _verify_folder(config: RerankerConfig, folder: Path) -> dict:
    try:
        manifest = json.loads((folder / "retrievallab-model.json").read_text(encoding="utf-8"))
        if (manifest["model_id"], manifest["revision"]) != (config.model_id, config.revision):
            raise DatasetError("Reranker model identity differs from config")
        if set(manifest["file_sha256"]) != set(MODEL_FILES):
            raise DatasetError("Reranker manifest has unexpected files")
        for name in MODEL_FILES:
            if sha256_file(folder / name) != manifest["file_sha256"][name]:
                raise DatasetError(f"Reranker file checksum mismatch: {name}")
        if (
            config.revision == RerankerConfig().revision
            and manifest["file_sha256"]["model.safetensors"] != WEIGHT_SHA256
        ):
            raise DatasetError("Reranker weights differ from the declared upstream artifact")
        architecture = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if (
            architecture["architectures"] != ["BertForSequenceClassification"]
            or architecture["num_hidden_layers"] != 2
            or architecture["hidden_size"] != 128
            or len(architecture["id2label"]) != 1
        ):
            raise DatasetError("Unexpected TinyBERT architecture")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(
            f"Verified reranker unavailable. Run: retrievallab reranker-download. {exc}"
        ) from exc


def verify_model(config: RerankerConfig, root: Path) -> dict:
    if config.checkpoint_sha256:
        from retrieval_lab.training import verify_checkpoint

        return verify_checkpoint(model_folder(config, root), config)
    return _verify_folder(config, model_folder(config, root))


def download_model(config: RerankerConfig, root: Path = Path("models")) -> Path:
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
            raise DatasetError("Hub returned a different reranker revision")
        blobs = {row["rfilename"]: row for row in upstream["siblings"]}
        with tempfile.TemporaryDirectory(prefix=".reranker-", dir=root) as temp:
            stage = Path(temp) / "model"
            stage.mkdir()
            hashes = {}
            for name in MODEL_FILES:
                info = blobs[name]
                if info["size"] > 25 * 1024**2:
                    raise DatasetError("Unexpectedly large TinyBERT asset")
                path = stage / name
                _fetch_file(
                    f"https://huggingface.co/{config.model_id}/resolve/{config.revision}/{name}",
                    path,
                    info["size"],
                    context,
                )
                size = path.stat().st_size
                if size != info["size"]:
                    raise DatasetError(f"Incomplete reranker asset: {name}")
                hashes[name] = sha256_file(path)
                if info.get("lfs"):
                    if hashes[name] != info["lfs"]["sha256"]:
                        raise DatasetError(f"Reranker upstream SHA-256 mismatch: {name}")
                else:
                    digest = hashlib.sha1(
                        f"blob {size}\0".encode() + path.read_bytes(), usedforsecurity=False
                    )
                    if digest.hexdigest() != info["blobId"]:
                        raise DatasetError(f"Reranker upstream Git-blob mismatch: {name}")
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
        return target
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not download pinned reranker: {exc}") from exc

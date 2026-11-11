"""Package local source, checked experiment evidence and trained checkpoints; no raw dataset."""

import json
import zipfile
from pathlib import Path

from retrieval_lab.data import sha256_file


def main():
    root = Path.cwd()
    artifact = root / "artifacts/RetrievalLab-1.0.0.zip"
    manifest_path = root / "artifacts/RetrievalLab-1.0.0-manifest.json"
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError("Release package exists; preserve it and use a new version")
    if not (root / "runs/m8-official-test-v1/result.json").exists():
        raise ValueError("Complete the official test before packaging the release")
    files = set()

    def add_tree(folder):
        files.update(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
            and not any(part.endswith(".egg-info") for part in p.parts)
        )

    for name in [
        "src",
        "tests",
        "docs",
        "configs",
        "scripts",
        ".github",
        "releases",
        "comparisons",
    ]:
        add_tree(root / name)
    for folder in (root / "runs").iterdir():
        if folder.is_dir() and (
            (folder / "manifest.json").exists()
            or folder.name in {"m7-training-v1", "m8-official-test-v1", "m4-configs", "m5-configs"}
        ):
            add_tree(folder)
    for name in [
        "README.md",
        "pyproject.toml",
        "app.py",
        ".gitignore",
        "requirements.lock.txt",
        "requirements-dense.lock.txt",
        "requirements-release.lock.txt",
        "dist/retrievallab-1.0.0-py3-none-any.whl",
        "runs/final-tests.log",
        "runs/final-tests.xml",
        "runs/m6-api-smoke-v1.json",
        "runs/m7-training.log",
        "runs/m7-validation.log",
        "runs/m7-comparison-pretrained-vs-trained.json",
    ]:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
    training = json.loads((root / "runs/m7-training-v1/result.json").read_text(encoding="utf-8"))
    for row in training["history"]:
        if "checkpoint_sha256" in row:
            add_tree(root / "models/checkpoints" / row["checkpoint_sha256"])
    identities = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}
    artifact.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(artifact) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(identities):
            raise ValueError("Package integrity verification failed")
    manifest = {
        "version": "1.0.0",
        "files": len(files),
        "size_bytes": artifact.stat().st_size,
        "zip_sha256": sha256_file(artifact),
        "file_sha256": identities,
        "excluded": [
            "virtual environment",
            "raw SciFact data",
            "pretrained model assets",
            "dense indexes",
            "synced sources",
            "temporary test directories",
        ],
        "setup": "Download verified pretrained assets/data with the documented commands. "
        "Trained safetensors are included. Install the pinned reference environment. "
        "Original freeze is strict about environment/hardware identity.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()

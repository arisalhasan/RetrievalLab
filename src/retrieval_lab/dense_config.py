"""A fully specified CPU embedding baseline; no model downloads on import."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DenseConfig:
    schema_version: int = 1
    system: str = "dense"
    implementation_version: str = "1.0.0"
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dimension: int = 384
    max_seq_length: int = 256
    batch_size: int = 32
    normalize_embeddings: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    pooling: str = "mean"
    index_type: str = "IndexFlatIP"
    document_fields: tuple[str, ...] = ("title", "text")
    query_prompt: str = ""
    document_prompt: str = ""
    torch_threads: int = 1
    faiss_threads: int = 1
    seed: int = 42

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Dense models require an immutable 40-character commit revision")
        if self.model_id != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("Milestone 3 supports the declared MiniLM baseline only")
        fixed = {
            "schema_version": 1,
            "system": "dense",
            "implementation_version": "1.0.0",
            "dimension": 384,
            "normalize_embeddings": True,
            "dtype": "float32",
            "device": "cpu",
            "pooling": "mean",
            "index_type": "IndexFlatIP",
            "document_fields": ("title", "text"),
            "query_prompt": "",
            "document_prompt": "",
        }
        for key, value in fixed.items():
            if getattr(self, key) != value:
                raise ValueError(f"Unsupported dense setting: {key}")
        for key in ["max_seq_length", "batch_size", "torch_threads", "faiss_threads"]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_seq_length > 256:
            raise ValueError("This model baseline is limited to 256 tokens")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    @classmethod
    def from_file(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "document_fields" in values:
            values["document_fields"] = tuple(values["document_fields"])
        return cls(**values)

    def to_dict(self) -> dict:
        return {**asdict(self), "document_fields": list(self.document_fields)}

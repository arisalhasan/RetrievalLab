"""The sole neural boundary: verified local files to normalized CPU embeddings."""

import importlib.metadata
import os
import threading
from pathlib import Path

import numpy as np

from retrieval_lab.dense_config import DenseConfig
from retrieval_lab.model_store import model_folder, verify_model


class SentenceEncoder:
    def __init__(self, config: DenseConfig, model_root: Path):
        manifest = verify_model(config, model_root)
        # Offline inference is deliberate. Downloading is a separate explicit command.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config.torch_threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.lock = threading.RLock()
        self.model = SentenceTransformer(
            str(model_folder(config, model_root).resolve()),
            device="cpu",
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            prompts={"query": "", "document": ""},
            model_kwargs={
                "use_safetensors": True,
                "torch_dtype": torch.float32,
                "attn_implementation": "eager",
            },
        )
        self.model.max_seq_length = config.max_seq_length
        self.model.eval()
        if self.model.get_sentence_embedding_dimension() != config.dimension:
            raise ValueError("Loaded model dimension does not match the dense config")
        self.fingerprint = {
            "model_id": config.model_id,
            "revision": config.revision,
            "model_file_sha256": manifest["file_sha256"],
            "libraries": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "safetensors",
                    "faiss-cpu",
                ]
            },
            "backend": "PyTorch CPU float32 eager attention",
            "trust_remote_code": False,
            "pooling": "attention-mask mean pooling followed by normalization",
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with self.lock, torch.inference_mode():
            result = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
                prompt="",
                device="cpu",
            )
        return np.ascontiguousarray(result, dtype=np.float32)

    def token_statistics(self, texts: list[str]) -> dict:
        lengths = []
        with self.lock:
            for start in range(0, len(texts), self.config.batch_size):
                batch = self.model.tokenizer(
                    texts[start : start + self.config.batch_size],
                    padding=False,
                    truncation=False,
                    return_length=True,
                    verbose=False,
                )
                lengths.extend(batch["length"])
        return {
            "texts": len(texts),
            "max_input_tokens": max(lengths, default=0),
            "texts_truncated": sum(length > self.config.max_seq_length for length in lengths),
            "max_seq_length": self.config.max_seq_length,
            "policy": "right truncation; lengths include special tokens",
        }

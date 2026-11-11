"""Local CPU pair scoring. Raw logits rank relevance; they are not truth probabilities."""

import importlib.metadata
import os
import threading
from pathlib import Path

import numpy as np

from retrieval_lab.reranker_store import model_folder, verify_model
from retrieval_lab.reranking import RerankerConfig


class CrossEncoderScorer:
    def __init__(self, config: RerankerConfig, model_root: Path):
        manifest = verify_model(config, model_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(config.torch_threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.config, self.lock = config, threading.RLock()
        self.model = CrossEncoder(
            str(model_folder(config, model_root).resolve()),
            num_labels=1,
            max_length=config.max_length,
            activation_fn=torch.nn.Identity(),
            device="cpu",
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            model_kwargs={
                "use_safetensors": True,
                "dtype": torch.float32,
                "attn_implementation": "eager",
            },
        )
        self.model.eval()
        self.fingerprint = {
            "model_id": config.model_id,
            "revision": config.revision,
            "model_file_sha256": manifest["file_sha256"],
            "activation": config.activation,
            "libraries": {
                n: importlib.metadata.version(n)
                for n in [
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "safetensors",
                ]
            },
            "backend": "PyTorch CPU float32 eager attention",
            "trust_remote_code": False,
        }

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        with self.lock:
            values = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                apply_softmax=False,
                convert_to_numpy=True,
            )
        values = np.asarray(values)
        if values.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError("Cross-encoder returned invalid scalar logits")
        return [float(score) for score in values]

    def token_statistics(self, pairs: list[tuple[str, str]]) -> dict:
        lengths = []
        with self.lock:
            for start in range(0, len(pairs), self.config.batch_size):
                batch = self.model.tokenizer(
                    pairs[start : start + self.config.batch_size],
                    padding=False,
                    truncation=False,
                    return_length=True,
                    verbose=False,
                )
                lengths.extend(batch["length"])
        return {
            "pairs": len(pairs),
            "max_joint_input_tokens": max(lengths, default=0),
            "pairs_truncated": sum(n > self.config.max_length for n in lengths),
            "max_length": self.config.max_length,
            "policy": "longest_first; includes special tokens",
        }

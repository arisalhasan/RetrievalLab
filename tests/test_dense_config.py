import json
from pathlib import Path

import pytest

from retrieval_lab.dense_config import DenseConfig


def test_pinned_dense_config_is_json_roundtrip_safe():
    root = Path(__file__).resolve().parents[1]
    config = DenseConfig.from_file(root / "configs/dense.json")
    assert config == DenseConfig()
    assert config.to_dict() == json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revision": "main"},
        {"device": "cuda"},
        {"normalize_embeddings": False},
        {"dtype": "float16"},
        {"dimension": 768},
        {"max_seq_length": 512},
        {"batch_size": 0},
        {"torch_threads": True},
        {"seed": True},
    ],
)
def test_dense_config_rejects_unsupported_or_ambiguous_experiments(kwargs):
    with pytest.raises(ValueError):
        DenseConfig(**kwargs)

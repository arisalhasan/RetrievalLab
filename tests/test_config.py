import json
from pathlib import Path

import pytest

from retrieval_lab.bm25 import BM25Config

CONFIG = Path(__file__).resolve().parents[1] / "configs/bm25.json"


def test_recorded_default_config_loads():
    assert BM25Config.from_file(CONFIG) == BM25Config()


def test_config_survives_json_roundtrip_without_changing_types():
    config = BM25Config().to_dict()
    assert json.loads(json.dumps(config)) == config


@pytest.mark.parametrize("setting", ["tokenizer", "implementation_version", "idf"])
def test_config_cannot_mislabel_implementation(setting, tmp_path):
    values = json.loads(CONFIG.read_text(encoding="utf-8"))
    values[setting] = "not-implemented"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported BM25 setting"):
        BM25Config.from_file(path)

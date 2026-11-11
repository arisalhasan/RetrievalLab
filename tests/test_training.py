import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from retrieval_lab.data import DatasetError, load_dataset
from retrieval_lab.reranking import RerankerConfig
from retrieval_lab.splits import make_split
from retrieval_lab.training import TrainingConfig, prepare_pairs, select_epoch, verify_checkpoint


def test_training_pairs_exclude_validation_and_test(mini_dataset):
    with (mini_dataset / "queries.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"_id": "train-2", "text": "planets"}) + "\n")
    with (mini_dataset / "qrels/train.tsv").open("a", encoding="utf-8") as stream:
        stream.write("train-2\td2\t1\n")
    (mini_dataset / "qrels/test.tsv").write_text("DO NOT READ", encoding="utf-8")
    dataset = load_dataset(mini_dataset)
    split = make_split(dataset)
    rows = prepare_pairs(dataset, split, TrainingConfig())
    assert rows == prepare_pairs(dataset, split, TrainingConfig())
    assert {r["query_id"] for r in rows} == set(split["train_query_ids"])
    assert not {r["query_id"] for r in rows} & set(split["validation_query_ids"])
    for row in rows:
        assert row["label"] == int(dataset.qrels[row["query_id"]].get(row["doc_id"], 0) > 0)
    bad = {**split, "train_query_ids": split["validation_query_ids"]}
    with pytest.raises(DatasetError, match="overlap"):
        prepare_pairs(dataset, bad, TrainingConfig())


def test_selection_includes_pretrained_and_ties_choose_earlier():
    rows = [{"epoch": n, "ndcg": 0.7, "recall": 0.8} for n in [2, 0, 1]]
    assert select_epoch(rows)["epoch"] == 0
    rows[0]["ndcg"] = 0.71
    assert select_epoch(rows)["epoch"] == 2
    with pytest.raises(ValueError):
        select_epoch([{"ndcg": float("nan"), "recall": 1}])


@pytest.mark.parametrize(
    "settings",
    [
        {"epochs": 0},
        {"epochs": 6},
        {"batch_size": True},
        {"learning_rate": float("inf")},
        {"seed": -1},
    ],
)
def test_invalid_training_config(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


def test_checkpoint_identity_and_missing_assets(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        replace(RerankerConfig(), checkpoint_sha256="../escape")
    config = replace(RerankerConfig(), checkpoint_sha256="a" * 64)
    assert RerankerConfig.from_dict(config.to_dict()) == config
    with pytest.raises(DatasetError, match="unavailable"):
        verify_checkpoint(Path(tmp_path), config)


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("RUN_RERANKER_MODEL_TESTS") != "1", reason="Opt-in real training"
)
def test_real_training_saves_and_reloads_safe_checkpoint(mini_dataset, tmp_path):
    from retrieval_lab.cross_encoder import CrossEncoderScorer
    from retrieval_lab.splits import write_split
    from retrieval_lab.training import train

    root = Path(__file__).resolve().parents[1]
    with (mini_dataset / "queries.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"_id": "train-2", "text": "new cells"}) + "\n")
    with (mini_dataset / "qrels/train.tsv").open("a", encoding="utf-8") as stream:
        stream.write("train-2\td3\t1\n")
    (mini_dataset / "qrels/test.tsv").write_text("DO NOT READ", encoding="utf-8")
    split = tmp_path / "split.json"
    write_split(make_split(load_dataset(mini_dataset)), split)
    config = tmp_path / "training.json"
    config.write_text(json.dumps({"epochs": 1}))
    # Parent assets are read-only; checkpoints are confined to the test's own model root.
    import shutil

    from retrieval_lab.reranker_store import model_folder

    model_root = tmp_path / "models"
    parent = RerankerConfig()
    shutil.copytree(model_folder(parent, root / "models"), model_folder(parent, model_root))
    from retrieval_lab.dense_config import DenseConfig
    from retrieval_lab.model_store import model_folder as dense_folder

    shutil.copytree(
        dense_folder(DenseConfig(), root / "models"), dense_folder(DenseConfig(), model_root)
    )
    output = tmp_path / "training-run"
    result = train(
        mini_dataset,
        split,
        root / "configs/reranker.json",
        config,
        output,
        model_root=model_root,
        cache_root=tmp_path / "indexes",
    )
    assert result["test_labels_read"] is False
    assert result["pair_count"] == 3
    trained = RerankerConfig.from_file(output / "best-trained-config.json")
    assert trained.checkpoint_sha256 is not None
    assert trained.checkpoint_sha256 != "a" * 64
    scorer = CrossEncoderScorer(trained, model_root)
    assert len(scorer.score([("cells", "cell biology")])) == 1
    assert all(row["reference_check"]["passed"] for row in result["history"])

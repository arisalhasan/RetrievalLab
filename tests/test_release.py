import json
import shutil
from pathlib import Path

import pytest

from retrieval_lab.bm25 import BM25Config
from retrieval_lab.data import DatasetError, load_dataset
from retrieval_lab.evaluation import EvaluationConfig, evaluate, load_report, run_evaluation
from retrieval_lab.release import (
    freeze,
    run_final,
    verify_freeze,
)
from retrieval_lab.release import (
    test_partition as official_partition,
)
from retrieval_lab.splits import make_split, write_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_project(mini_dataset, tmp_path):
    with (mini_dataset / "queries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "train-2", "text": "planets"}) + "\n")
    with (mini_dataset / "qrels/train.tsv").open("a", encoding="utf-8") as f:
        f.write("train-2\td2\t1\n")
    root = tmp_path / "project"
    data = root / "data/scifact"
    shutil.copytree(mini_dataset, data)
    (root / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/bm25.json", root / "configs/bm25.json")
    config = root / "configs/evaluation.json"
    config.write_text(
        json.dumps({"repetitions": 1, "bootstrap_samples": 10, "latency_query_limit": 1})
    )
    split = root / "configs/split.json"
    write_split(make_split(load_dataset(data)), split)
    run_evaluation(data, split, root / "configs/bm25.json", config, root / "runs/validation")
    spec = root / "configs/release.json"
    spec.write_text(
        json.dumps(
            {
                "systems": {
                    "bm25": {
                        "system": "bm25",
                        "config": "configs/bm25.json",
                        "validation_run": "runs/validation",
                    }
                },
                "evaluation_config": "configs/evaluation.json",
            }
        )
    )
    return root, spec, root / "release/freeze.json"


def test_freeze_does_not_parse_test_and_final_requires_unchanged_inputs(release_project):
    root, spec, path = release_project
    test_file = root / "data/scifact/qrels/test.tsv"
    test_file.write_text("BROKEN TEST LABELS")
    manifest = freeze(spec, path, root)
    assert manifest["test_labels_parsed_at_freeze"] is False
    assert verify_freeze(path, root) == manifest
    with pytest.raises(DatasetError, match="header"):
        run_final(path, root / "runs/final", root)
    assert (root / "runs/final/test-access.jsonl").exists()
    with pytest.raises(FileExistsError):
        freeze(spec, path, root)
    config = root / "configs/bm25.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(DatasetError, match="input changed"):
        verify_freeze(path, root)


def test_official_evaluation_and_resume_preserve_outputs(release_project):
    root, spec, path = release_project
    freeze(spec, path, root)
    output = root / "runs/final"
    result = run_final(path, output, root)
    assert result["selection_changed_after_test"] is False
    dataset = load_dataset(root / "data/scifact", "test")
    summary, rows, timings = load_report(output / "bm25", dataset)
    assert summary["run_type"] == "official_test_evaluation"
    assert rows[0]["query_id"] == "test-1"
    assert summary["quality"]["ndcg"] == 1
    original = (output / "bm25/manifest.json").read_bytes()
    assert run_final(path, output, root) == result
    assert (output / "bm25/manifest.json").read_bytes() == original
    from retrieval_lab.final_report import export_release

    exported = root / "export"
    values = export_release(path, output, exported, root)
    assert values[0]["ndcg_at_10"] == 1
    assert (exported / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert "Selected before test: bm25" in (exported / "OFFICIAL_TEST_V1.md").read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        export_release(path, output, exported, root)
    with pytest.raises(DatasetError, match="requires a verified release freeze"):
        evaluate(dataset, official_partition(dataset, "a" * 64), BM25Config(), EvaluationConfig())


def test_freeze_tamper_rejected(release_project):
    root, spec, path = release_project
    manifest = freeze(spec, path, root)
    manifest["selected_system"] = "invented"
    path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="freeze checksum"):
        verify_freeze(path, root)


def test_isolated_worker_and_final_dashboard(release_project):
    from streamlit.testing.v1 import AppTest

    root, spec, path = release_project
    freeze(spec, path, root)
    output = root / "runs/m8-official-test-v1"
    result = run_final(path, output, root, isolated_processes=True)
    assert result["quality"]["bm25"]["ndcg"] == 1
    expected_freeze = root / "releases/v1/freeze.json"
    expected_freeze.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, expected_freeze)
    script = (
        "from pathlib import Path\n"
        "from retrieval_lab.release_dashboard import render_final\n"
        f"render_final(Path({str(root)!r}), Path({str(root / 'data/scifact')!r}))"
    )
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert any("Selected before test: bm25" in item.value for item in app.subheader)
    manifest_path = output / "bm25/manifest.json"
    corrupt = json.loads(manifest_path.read_text())
    corrupt["freeze_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(corrupt))
    app.run()
    assert not app.exception
    assert any("verification failed" in item.value for item in app.error)


@pytest.mark.parametrize("system", ["hybrid", "rerank"])
def test_official_neural_trace_reports_roundtrip(mini_dataset, tmp_path, monkeypatch, system):
    from test_dense import SyntheticEncoder
    from test_reranking import SyntheticPairScorer

    from retrieval_lab import cross_encoder, encoders
    from retrieval_lab.evaluation import save_evaluation
    from retrieval_lab.hybrid import HybridConfig
    from retrieval_lab.reranking import RerankerConfig

    monkeypatch.setattr(encoders, "SentenceEncoder", lambda config, root: SyntheticEncoder())
    monkeypatch.setattr(
        cross_encoder, "CrossEncoderScorer", lambda config, root: SyntheticPairScorer()
    )
    dataset = load_dataset(mini_dataset, "test")
    identity = "a" * 64
    split = official_partition(dataset, identity)
    config = HybridConfig() if system == "hybrid" else RerankerConfig()
    summary, rows, timings = evaluate(
        dataset,
        split,
        config,
        EvaluationConfig(repetitions=1, bootstrap_samples=10, latency_query_limit=1),
        cache_root=tmp_path / "indexes",
        official_freeze_sha256=identity,
    )
    output = tmp_path / "official"
    save_evaluation(summary, rows, timings, split, output)
    assert load_report(output, dataset)[0] == summary
    analysis = (
        summary["candidate_analysis"] if system == "hybrid" else summary["reranking_analysis"]
    )
    assert "test" in analysis["scope"] and "validation" not in analysis["scope"]

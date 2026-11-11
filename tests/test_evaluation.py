import json
from pathlib import Path

import pytest

from retrieval_lab.cli import main
from retrieval_lab.data import DatasetError, load_dataset
from retrieval_lab.evaluation import (
    EvaluationConfig,
    cluster_intervals,
    load_report,
    run_evaluation,
)
from retrieval_lab.splits import make_split, write_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(mini_dataset, tmp_path):
    # Add independent training queries so the tiny integration fixture can be split.
    with (mini_dataset / "queries.jsonl").open("a", encoding="utf-8") as stream:
        for qid, text in [("train-2", "planets orbit"), ("train-3", "new cells")]:
            stream.write(json.dumps({"_id": qid, "text": text}) + "\n")
    with (mini_dataset / "qrels/train.tsv").open("a", encoding="utf-8") as stream:
        stream.write("train-2\td2\t1\ntrain-3\td3\t1\n")
    dataset = load_dataset(mini_dataset)
    split_path = tmp_path / "split.json"
    write_split(make_split(dataset), split_path)
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps({"warmup_passes": 1, "repetitions": 2, "bootstrap_samples": 20}),
        encoding="utf-8",
    )
    return mini_dataset, split_path, config, tmp_path / "run"


def test_complete_experiment_and_report_integrity(experiment):
    data, split, config, output = experiment
    # Test labels are deliberately unusable: the entire evaluation must ignore them.
    (data / "qrels/test.tsv").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    report = run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    summary, queries, timings = load_report(output, load_dataset(data))
    assert summary == report
    assert summary["dataset"]["partition"] == "validation"
    assert summary["reference_check"]["passed"]
    assert len(timings) == len(queries) * 2
    assert summary["quality"]["recall"] == 1.0
    assert summary["resources"]["measured_queries"]["sampled_peak_rss_bytes"] > 0
    with pytest.raises(FileExistsError):
        run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    (output / "queries.json").write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetError, match="checksum mismatch"):
        load_report(output, load_dataset(data))


def test_cli_split_verification_and_evaluation(experiment, tmp_path, capsys):
    data, split, config, output = experiment
    generated = tmp_path / "another-split.json"
    assert main(["split", "--data-dir", str(data), "--output", str(generated)]) == 0
    assert main(["verify-split", "--data-dir", str(data), "--manifest", str(generated)]) == 0
    assert (
        main(
            [
                "evaluate",
                "--data-dir",
                str(data),
                "--split-manifest",
                str(split),
                "--evaluation-config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert '"passed": true' in capsys.readouterr().out


def test_report_rejects_an_edited_aggregate(experiment):
    data, split, config, output = experiment
    run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["quality"]["recall"] = 0.123456
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetError, match="per-query evidence"):
        load_report(output, load_dataset(data))


def test_cluster_bootstrap_is_seeded_and_preserves_group_membership():
    scores = {"a": {"recall": 0.0, "ndcg": 0.0}, "b": {"recall": 1.0, "ndcg": 1.0}}
    groups = {"same": ["a", "b"]}
    result = cluster_intervals(scores, groups, seed=42, samples=20)
    assert result["recall"] == [0.5, 0.5]  # Resampling a group never separates its members.
    assert result == cluster_intervals(scores, groups, seed=42, samples=20)


def test_timing_sample_never_drops_quality_queries(experiment, tmp_path):
    from dataclasses import asdict

    data, _, _, output = experiment
    with (
        (data / "queries.jsonl").open("a", encoding="utf-8") as queries,
        (data / "corpus.jsonl").open("a", encoding="utf-8") as corpus,
        (data / "qrels/train.tsv").open("a", encoding="utf-8") as labels,
    ):
        for i in range(4, 10):
            queries.write(json.dumps({"_id": f"q{i}", "text": f"uniqueterm{i}"}) + "\n")
            corpus.write(
                json.dumps({"_id": f"d{i}", "title": f"uniqueterm{i}", "text": "abstract"}) + "\n"
            )
            labels.write(f"q{i}\td{i}\t1\n")
    dataset = load_dataset(data)
    split = tmp_path / "larger-split.json"
    write_split(make_split(dataset, validation_fraction=0.5), split)
    config = tmp_path / "sampled.json"
    config.write_text(
        json.dumps(
            asdict(EvaluationConfig(latency_query_limit=1, repetitions=2, bootstrap_samples=20))
        ),
        encoding="utf-8",
    )
    report = run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    _, rows, timings = load_report(output, dataset)
    assert len(rows) >= 3 and len(rows) == report["quality"]["query_count"]
    assert len(timings) == 2 and len({row["query_id"] for row in timings}) == 1
    assert len(report["measurement_protocol"]["latency_query_ids"]) == 1


def test_rerank_experiment_diagnostics_and_ui(experiment, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    from streamlit.testing.v1 import AppTest
    from test_dense import SyntheticEncoder
    from test_reranking import SyntheticPairScorer

    monkeypatch.setattr("retrieval_lab.encoders.SentenceEncoder", lambda *args: SyntheticEncoder())
    monkeypatch.setattr(
        "retrieval_lab.cross_encoder.CrossEncoderScorer", lambda *args: SyntheticPairScorer()
    )
    data, split, config, output = experiment
    (data / "qrels/test.tsv").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    report = run_evaluation(
        data,
        split,
        ROOT / "configs/reranker.json",
        config,
        output,
        system="rerank",
        cache_root=tmp_path / "cache",
    )
    summary, rows, timings = load_report(output, load_dataset(data))
    assert summary == report and report["reference_check"]["passed"]
    assert report["reranking_analysis"]["candidate_recall"] >= report["quality"]["recall"]
    assert len(timings) == 2 * len(rows)
    for row in rows:
        assert {r["doc_id"] for r in row["results"]} <= {
            r["doc_id"] for r in row["reranking_trace"]["candidates"]
        }
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("TinyBERT" in item.value for item in app.subheader)
    app.selectbox[1].select("All validation queries").run()
    assert any(
        item.label == "Inspect reranking changes and candidate misses" for item in app.expander
    )
    path = output / "manifest.json"
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["reranking_analysis"]["candidate_recall"] = -1
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(DatasetError, match="Reranking summary"):
        load_report(output, load_dataset(data))


def test_dense_evaluation_and_paired_dashboard(experiment, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    from streamlit.testing.v1 import AppTest
    from test_dense import SyntheticEncoder

    from retrieval_lab.comparison import compare_runs

    monkeypatch.setattr(
        "retrieval_lab.encoders.SentenceEncoder", lambda config, root: SyntheticEncoder()
    )
    data, split, config, baseline_dir = experiment
    # Malformed test labels prove neither retriever nor comparison reads them.
    (data / "qrels/test.tsv").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    run_evaluation(data, split, ROOT / "configs/bm25.json", config, baseline_dir)
    candidate_dir = tmp_path / "dense"
    report = run_evaluation(
        data,
        split,
        ROOT / "configs/dense.json",
        config,
        candidate_dir,
        system="dense",
        cache_root=tmp_path / "cache",
    )
    assert report["exact_search_check"]["passed"]
    assert report["retriever_preparation"]["identity"]["encoder"]["backend"].startswith("synthetic")
    assert (
        len(load_report(candidate_dir, load_dataset(data))[2])
        == 2 * report["quality"]["query_count"]
    )
    compared = compare_runs(baseline_dir, candidate_dir, data)
    assert compared["query_count"] == report["quality"]["query_count"]
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(button for button in app.button if button.label == "Compare runs").click().run()
    assert not app.exception
    assert any("Paired comparison" in item.value for item in app.subheader)
    assert any("Candidate minus baseline" in item.value for item in app.markdown)


def test_hybrid_cli_evaluation_candidate_evidence_and_dashboard(experiment, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    from streamlit.testing.v1 import AppTest
    from test_dense import SyntheticEncoder

    monkeypatch.setattr(
        "retrieval_lab.encoders.SentenceEncoder", lambda config, root: SyntheticEncoder()
    )
    data, split, config, output = experiment
    (data / "qrels/test.tsv").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    assert (
        main(
            [
                "evaluate",
                "--system",
                "hybrid",
                "--data-dir",
                str(data),
                "--split-manifest",
                str(split),
                "--evaluation-config",
                str(config),
                "--output",
                str(output),
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    report, rows, timings = load_report(output, load_dataset(data))
    assert report["system"] == "Hybrid (RRF)"
    assert report["reference_check"]["passed"] and report["exact_search_check"]["passed"]
    assert report["candidate_analysis"]["union_recall"] >= report["quality"]["recall"]
    assert report["candidate_analysis"]["depth_per_source"] == 100
    assert len(timings) == 2 * len(rows)
    for row in rows:
        candidate = row["candidate_analysis"]
        assert candidate["union_recall"] >= max(candidate["source_recall"].values())
        assert candidate["relevant_dropped_by_fusion"] == []  # Only 3 corpus documents.
        for hit in row["fusion_trace"]["results"]:
            assert hit["rrf_score"] == pytest.approx(
                sum(part["contribution"] for part in hit["sources"].values())
            )
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Hybrid (RRF)" in item.value for item in app.subheader)
    app.selectbox[1].select("All validation queries").run()
    assert not app.exception
    assert any(item.label == "Explain fusion and candidate misses" for item in app.expander)
    path = output / "manifest.json"
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["candidate_analysis"]["union_recall"] = -1
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(DatasetError, match="candidate summary"):
        load_report(output, load_dataset(data))


def test_paired_comparison_protects_outputs_and_rejects_incompatible_runs(experiment, tmp_path):
    import copy

    from retrieval_lab.comparison import compare_loaded, compare_runs

    data, split_path, config, baseline_dir = experiment
    candidate_dir = tmp_path / "candidate"
    baseline = run_evaluation(data, split_path, ROOT / "configs/bm25.json", config, baseline_dir)
    candidate = run_evaluation(data, split_path, ROOT / "configs/bm25.json", config, candidate_dir)
    output = tmp_path / "comparison.json"
    compared = compare_runs(baseline_dir, candidate_dir, data, output)
    assert compared["candidate_minus_baseline"] == {"recall": 0.0, "ndcg": 0.0}
    assert compared["paired_95pct_intervals"]["ndcg"] == [0.0, 0.0]
    assert len(compared["per_query"]) == baseline["quality"]["query_count"]
    with pytest.raises(FileExistsError):
        compare_runs(baseline_dir, candidate_dir, data, output)
    rows = json.loads((baseline_dir / "queries.json").read_text())
    split = json.loads(split_path.read_text())
    changed = copy.deepcopy(candidate)
    changed["config"]["evaluation"]["repetitions"] += 1
    with pytest.raises(DatasetError, match="same evaluation configuration"):
        compare_loaded(baseline, rows, changed, rows, split)
    changed = copy.deepcopy(candidate)
    changed["split_sha256"] = "different-split"
    with pytest.raises(DatasetError, match="different split_sha256"):
        compare_loaded(baseline, rows, changed, rows, split)
    with pytest.raises(DatasetError, match="query IDs"):
        compare_loaded(baseline, rows, candidate, [], split)


def test_validation_dashboard_shows_real_report_and_handles_corruption(experiment, monkeypatch):
    from streamlit.testing.v1 import AppTest

    data, split, config, output = experiment
    run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(output.parent))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert {metric.label for metric in app.metric} >= {"Recall@10", "nDCG@10", "Query p95"}
    app.selectbox[1].select("All validation queries").run()
    assert not app.exception
    assert any("Retrieved ranking" in item.value for item in app.markdown)
    (output / "queries.json").write_text("[]", encoding="utf-8")
    app.run()
    assert not app.exception
    assert any("could not be verified" in message.value for message in app.error)


def test_failure_dashboard_shows_missed_abstract_when_no_documents_match(experiment, monkeypatch):
    from streamlit.testing.v1 import AppTest

    data, _, config, output = experiment
    path = data / "queries.jsonl"
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in queries:
        if row["_id"].startswith("train-"):
            row["text"] = "zzunmatched" + row["_id"].replace("-", "")
    path.write_text("".join(json.dumps(row) + "\n" for row in queries), encoding="utf-8")
    dataset = load_dataset(data)
    split = output.parent / "unmatched-split.json"
    write_split(make_split(dataset), split)
    run_evaluation(data, split, ROOT / "configs/bm25.json", config, output)
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(output.parent))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Recall@10") == "0.0000"
    _, rows, _ = load_report(output, dataset)
    missed = dataset.corpus[rows[0]["missed_relevant_ids"][0]]
    assert any(item.label == f"{missed.title} · {missed.doc_id}" for item in app.expander)
    assert any(item.value == missed.text for item in app.markdown)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_passes": 0},
        {"repetitions": -1},
        {"cutoff": 20},
        {"rss_sample_interval_ms": 0},
        {"seed": True},
        {"latency_query_limit": 0},
        {"latency_query_limit": True},
        {"latency_query_limit": -1},
    ],
)
def test_invalid_evaluation_settings(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)

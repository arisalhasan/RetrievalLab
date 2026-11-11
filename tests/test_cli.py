import json
from pathlib import Path

from retrieval_lab.cli import main

CONFIG = Path(__file__).resolve().parents[1] / "configs/bm25.json"


def test_cli_search_records_provenance_and_protects_previous_runs(mini_dataset, tmp_path, capsys):
    output = tmp_path / "run.json"
    args = [
        "search",
        "--data-dir",
        str(mini_dataset),
        "--config",
        str(CONFIG),
        "--query",
        "planets",
        "--top-k",
        "1",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["results"][0]["doc_id"] == "d2"
    assert record["dataset"]["split"] == "train"
    assert "qrels/test.tsv" not in record["dataset"]["file_sha256"]
    assert record["config"]["seed"] == 42
    assert record["timing_ms"]["query_only"] >= 0
    assert "not_benchmark" in record["run_type"]
    assert main(args) == 1
    assert "Error" in capsys.readouterr().err
    assert json.loads(output.read_text())["created_at_utc"] == record["created_at_utc"]


def test_cli_invalid_query_is_actionable(mini_dataset, capsys):
    assert main(["search", "--data-dir", str(mini_dataset), "--query", "  "]) == 1
    assert "Query must contain" in capsys.readouterr().err

"""Small synthetic BEIR-format data, never presented as SciFact results."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "synthetic-beir"
    (folder / "qrels").mkdir(parents=True)
    docs = [
        {"_id": "d1", "title": "Cell biology", "text": "Cells divide by mitosis."},
        {"_id": "d2", "title": "Space science", "text": "Planets orbit stars."},
        {"_id": "d3", "title": "Cell division", "text": "Mitosis creates new cells."},
    ]
    queries = [
        {"_id": "train-1", "text": "cell mitosis"},
        {"_id": "test-1", "text": "planets"},
    ]
    for filename, rows in [("corpus.jsonl", docs), ("queries.jsonl", queries)]:
        (folder / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    header = "query-id\tcorpus-id\tscore\n"
    (folder / "qrels/train.tsv").write_text(header + "train-1\td1\t1\n", encoding="utf-8")
    (folder / "qrels/test.tsv").write_text(header + "test-1\td2\t1\n", encoding="utf-8")
    return folder

"""Deterministic, group-aware partitions of official training queries only."""

import hashlib
import json
import math
from pathlib import Path

from retrieval_lab.bm25 import tokenize
from retrieval_lab.data import Dataset, DatasetError

GROUPING_VERSION = "shared-positive-evidence-or-token-jaccard-v1"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _groups(dataset: Dataset, threshold: float) -> dict[str, list[str]]:
    """Connected components; transitive relationships must stay together."""
    ids = sorted(dataset.queries)
    parent = {qid: qid for qid in ids}

    def root(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    def join(left: str, right: str) -> None:
        a, b = sorted((root(left), root(right)))
        parent[b] = a

    evidence_owner: dict[str, str] = {}
    for qid in ids:
        for doc_id, grade in dataset.qrels[qid].items():
            if grade > 0:
                if doc_id in evidence_owner:
                    join(qid, evidence_owner[doc_id])
                evidence_owner[doc_id] = qid
    token_sets = {qid: set(tokenize(dataset.queries[qid])) for qid in ids}
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            a, b = token_sets[left], token_sets[right]
            union = a | b
            # Empty-token claims group by their normalized raw text, never by 0/0.
            similar = (
                (len(a & b) / len(union) >= threshold)
                if union
                else (
                    dataset.queries[left].casefold().strip()
                    == dataset.queries[right].casefold().strip()
                )
            )
            if similar:
                join(left, right)
    components: dict[str, list[str]] = {}
    for qid in ids:
        components.setdefault(root(qid), []).append(qid)
    return {canonical_hash({"query_ids": members}): members for members in components.values()}


def make_split(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    seed: int = 42,
    near_duplicate_threshold: float = 0.8,
) -> dict:
    if dataset.split != "train":
        raise DatasetError("Derived splits must use official train judgments only")
    if set(dataset.queries) != set(dataset.qrels):
        raise DatasetError("Every source query must have a judgments entry")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if not math.isfinite(near_duplicate_threshold) or not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups = _groups(dataset, near_duplicate_threshold)
    if len(groups) < 2:
        raise DatasetError("At least two independent query groups are needed for a split")
    # Hash ordering is stable across Python versions and input row order.
    ordered = sorted(groups, key=lambda gid: canonical_hash({"seed": seed, "group": gid}))
    target = len(dataset.queries) * validation_fraction
    cumulative = 0
    prefixes = []
    for position, gid in enumerate(ordered[:-1], 1):
        cumulative += len(groups[gid])
        prefixes.append((abs(cumulative - target), position))
    _, boundary = min(prefixes)
    validation_ids = sorted(qid for gid in ordered[:boundary] for qid in groups[gid])
    training_ids = sorted(set(dataset.queries) - set(validation_ids))
    payload = {
        "schema_version": 1,
        "name": "scifact-train-derived-v1",
        "source_split": "train",
        "source_file_sha256": dataset.file_sha256,
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "grouping": {
            "version": GROUPING_VERSION,
            "token_jaccard_threshold": near_duplicate_threshold,
        },
        "groups": dict(sorted(groups.items())),
        "train_query_ids": training_ids,
        "validation_query_ids": validation_ids,
        "counts": {
            "source_queries": len(dataset.queries),
            "groups": len(groups),
            "largest_group": max(map(len, groups.values())),
            "train": len(training_ids),
            "validation": len(validation_ids),
        },
    }
    return {**payload, "split_sha256": canonical_hash(payload)}


def write_split(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")


def load_split(path: str | Path, dataset: Dataset) -> dict:
    """Regenerate the manifest: catch edited IDs, grouping, seed, and data drift."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = make_split(
            dataset,
            manifest["validation_fraction_target"],
            manifest["seed"],
            manifest["grouping"]["token_jaccard_threshold"],
        )
        if manifest != expected:
            raise DatasetError("Split does not match its source data, grouping rules, or checksum")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Invalid split manifest: {exc}") from exc

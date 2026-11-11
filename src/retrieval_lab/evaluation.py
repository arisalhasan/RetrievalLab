"""Comparable retrieval experiments on a frozen train-derived validation partition."""

import csv
import io
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

from retrieval_lab.bm25 import tokenize
from retrieval_lab.data import Dataset, DatasetError, load_dataset, sha256_file
from retrieval_lab.files import install_directory
from retrieval_lab.hybrid import HybridConfig
from retrieval_lab.metrics import METRIC_VERSION, aggregate, check_reference, score_query
from retrieval_lab.profiling import ResourceMonitor, latency_summary, percentile
from retrieval_lab.provenance import code_state, environment, source_hashes
from retrieval_lab.reranking import RerankerConfig
from retrieval_lab.retrievers import RetrieverConfig, build_retriever, load_config
from retrieval_lab.splits import load_split


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int = 1
    cutoff: int = 10
    warmup_passes: int = 1
    repetitions: int = 5
    seed: int = 42
    rss_sample_interval_ms: int = 10
    bootstrap_samples: int = 1000
    latency_query_limit: int | None = None

    def __post_init__(self):
        if self.schema_version != 1 or self.cutoff != 10:
            raise ValueError("This evaluation schema supports cutoff 10 only")
        for field in [
            "warmup_passes",
            "repetitions",
            "rss_sample_interval_ms",
            "bootstrap_samples",
        ]:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.latency_query_limit is not None and (
            isinstance(self.latency_query_limit, bool)
            or not isinstance(self.latency_query_limit, int)
            or self.latency_query_limit < 1
        ):
            raise ValueError("latency_query_limit must be a positive integer or null")

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def cluster_intervals(scores: dict, groups: dict, seed: int, samples: int) -> dict:
    """Resample whole validation groups, then take query-weighted macro averages."""
    selected = [members for members in groups.values() if members[0] in scores]
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {"recall": [], "ndcg": []}
    for _ in range(samples):
        picked = [qid for members in rng.choices(selected, k=len(selected)) for qid in members]
        for metric in distributions:
            distributions[metric].append(fmean(scores[qid][metric] for qid in picked))
    return {
        "method": "query-weighted cluster bootstrap over validation groups",
        "seed": seed,
        "samples": samples,
        "confidence": 0.95,
        "groups": len(selected),
        **{
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in distributions.items()
        },
    }


def analyze_candidates(trace: dict, qid: str, qrels: dict, final_ids: list[str]) -> dict:
    """Keep all relevance-label analysis outside the retriever. No-positive queries count zero."""
    positives = {did for did, grade in qrels.items() if grade > 0}
    source_ids = {name: set(ids) - {qid} for name, ids in trace["source_candidate_ids"].items()}
    union = set().union(*source_ids.values())
    found = union & positives
    return {
        "union_size_after_self_filter": len(union),
        "union_recall": len(found) / len(positives) if positives else 0.0,
        "source_recall": {
            name: len(ids & positives) / len(positives) if positives else 0.0
            for name, ids in source_ids.items()
        },
        "relevant_in_candidates": sorted(found),
        "relevant_missing_from_candidates": sorted(positives - union),
        "relevant_dropped_by_fusion": sorted(found - set(final_ids)),
    }


def summarize_candidates(rows: list[dict], depth: int, partition: str = "validation") -> dict:
    return {
        "scope": f"untimed post-measurement live retrieval; positive {partition} judgments; "
        "self IDs excluded; coverage is an upper bound, not final top-10 quality",
        "depth_per_source": depth,
        "mean_union_size": fmean(row["union_size_after_self_filter"] for row in rows),
        "union_recall": fmean(row["union_recall"] for row in rows),
        "source_recall": {
            name: fmean(row["source_recall"][name] for row in rows) for name in ["bm25", "dense"]
        },
        "queries_with_evidence_missing_from_candidates": sum(
            bool(row["relevant_missing_from_candidates"]) for row in rows
        ),
        "queries_with_candidate_evidence_dropped_from_top10": sum(
            bool(row["relevant_dropped_by_fusion"]) for row in rows
        ),
    }


def evaluate(
    dataset: Dataset,
    split: dict,
    retriever_config: RetrieverConfig,
    config: EvaluationConfig,
    model_root: Path = Path("models"),
    cache_root: Path = Path("indexes"),
    *,
    official_freeze_sha256: str | None = None,
) -> tuple:
    official = dataset.split == "test" and split["source_split"] == "test"
    if official and (
        official_freeze_sha256 is None or split.get("freeze_sha256") != official_freeze_sha256
    ):
        raise DatasetError("Official test evaluation requires a verified release freeze")
    if not official and (dataset.split != "train" or split["source_split"] != "train"):
        raise DatasetError(
            "Development experiments evaluate train-derived validation, never official test"
        )
    qids = split["validation_query_ids"]
    if not qids:
        raise ValueError("Validation partition is empty")
    if (
        isinstance(retriever_config, HybridConfig)
        and config.cutoff > retriever_config.candidate_depth
    ):
        raise ValueError("Hybrid candidate_depth must be at least the evaluation cutoff")
    if (
        isinstance(retriever_config, RerankerConfig)
        and config.cutoff > retriever_config.rerank_depth
    ):
        raise ValueError("rerank_depth must be at least the evaluation cutoff")
    with ResourceMonitor(config.rss_sample_interval_ms) as build:
        index = build_retriever(
            list(dataset.corpus.values()), retriever_config, model_root, cache_root
        )
    exact_check = (
        index.sanity_check([dataset.queries[qid] for qid in qids], config.cutoff)
        if retriever_config.system in {"dense", "hybrid", "rerank"}
        else None
    )
    ranked_hits, reranking_traces = {}, {}
    latency_qids = (
        qids
        if config.latency_query_limit is None
        else sorted(
            random.Random(config.seed).sample(qids, min(len(qids), config.latency_query_limit))
        )
    )

    def retrieve(qid, capture=False):
        if capture and isinstance(retriever_config, RerankerConfig):
            hits, trace = index.search_with_trace(dataset.queries[qid], config.cutoff)
            reranking_traces[qid] = trace
            return hits
        return index.search(dataset.queries[qid], config.cutoff)

    if config.latency_query_limit is not None:
        # Full quality coverage remains independent of the smaller timing sample.
        for qid in qids:
            ranked_hits[qid] = retrieve(qid, capture=True)
    for _ in range(config.warmup_passes):
        for qid in latency_qids:
            hits = retrieve(qid, capture=qid not in ranked_hits)
            if qid in ranked_hits and hits != ranked_hits[qid]:
                raise ValueError(f"Non-repeatable warmup ranking for query {qid}")
            ranked_hits[qid] = hits
    timings = []
    rng = random.Random(config.seed)
    with ResourceMonitor(config.rss_sample_interval_ms) as query_resources:
        for repetition in range(1, config.repetitions + 1):
            order = list(latency_qids)
            rng.shuffle(order)
            for qid in order:
                started = perf_counter()
                hits = index.search(dataset.queries[qid], config.cutoff)
                elapsed = (perf_counter() - started) * 1000
                if hits != ranked_hits[qid]:
                    raise ValueError(f"Non-repeatable ranking for query {qid}")
                timings.append({"repetition": repetition, "query_id": qid, "latency_ms": elapsed})
    scores, queries, rankings, candidate_rows = {}, [], {}, []
    group_for = {qid: gid for gid, members in split["groups"].items() for qid in members}
    for qid in qids:
        # Match BEIR's default: filter self IDs after the retriever's top-10 cutoff.
        hits = [hit for hit in ranked_hits[qid] if hit.document.doc_id != qid]
        rankings[qid] = [hit.document.doc_id for hit in hits]
        row = score_query(qid, rankings[qid], dataset.qrels[qid], config.cutoff)
        scores[qid] = row
        missed = sorted(
            did
            for did, grade in dataset.qrels[qid].items()
            if grade > 0 and did not in rankings[qid]
        )
        diagnostics = {}
        if isinstance(retriever_config, HybridConfig):
            # Extra live retrieval is untimed. Labels stay in the evaluator, never the index.
            traced_hits, trace = index.search_with_trace(dataset.queries[qid], config.cutoff)
            if traced_hits != ranked_hits[qid]:
                raise ValueError("Hybrid diagnostic ranking differs from measured ranking")
            candidate = analyze_candidates(trace, qid, dataset.qrels[qid], rankings[qid])
            candidate_rows.append(candidate)
            diagnostics = {"candidate_analysis": candidate, "fusion_trace": trace}
        if isinstance(retriever_config, RerankerConfig):
            from retrieval_lab.rerank_diagnostics import analyze

            trace = reranking_traces[qid]
            pairs = [
                (
                    dataset.queries[qid],
                    f"{dataset.corpus[c['doc_id']].title}\n{dataset.corpus[c['doc_id']].text}",
                )
                for c in trace["candidates"]
            ]
            trace["token_statistics"] = index.scorer.token_statistics(pairs)
            diagnostics = {
                "reranking_trace": trace,
                "reranking_analysis": analyze(trace, qid, dataset.qrels[qid], config.cutoff),
            }
        queries.append(
            {
                "query_id": qid,
                "query": dataset.queries[qid],
                "group_id": group_for[qid],
                "query_tokens": len(tokenize(dataset.queries[qid])),
                **row,
                **diagnostics,
                "missed_relevant_ids": missed,
                "results": [
                    {
                        "rank": n,
                        "doc_id": hit.document.doc_id,
                        "title": hit.document.title,
                        "score": hit.score,
                        "judgment": dataset.qrels[qid].get(hit.document.doc_id),
                    }
                    for n, hit in enumerate(hits, 1)
                ],
            }
        )
    reference = check_reference(
        {qid: dataset.qrels[qid] for qid in qids}, rankings, scores, config.cutoff
    )
    summary = {
        "schema_version": 2,
        "run_type": "official_test_evaluation"
        if official
        else "train_derived_validation_evaluation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "system": {
            "bm25": "BM25",
            "dense": "Dense (MiniLM)",
            "hybrid": "Hybrid (RRF)",
            "rerank": "Hybrid + TinyBERT",
        }[retriever_config.system],
        "dataset": {
            "name": "BEIR/SciFact",
            "source_split": dataset.split,
            "partition": "official_test" if official else "validation",
            "documents": len(dataset.corpus),
            "file_sha256": dataset.file_sha256,
        },
        "split_sha256": split["split_sha256"],
        "split_counts": split["counts"],
        "config": {"retriever": retriever_config.to_dict(), "evaluation": asdict(config)},
        "retriever_preparation": getattr(index, "preparation", {"implementation": "BM25"}),
        "exact_search_check": exact_check,
        "candidate_analysis": summarize_candidates(
            candidate_rows, retriever_config.candidate_depth, "test" if official else "validation"
        )
        if candidate_rows
        else None,
        "metric_definition": {
            "version": METRIC_VERSION,
            "cutoff": config.cutoff,
            "gain": "linear relevance grade",
            "aggregation": "query macro mean",
            "no_positive_labels": "zero, included",
            "unjudged_gain": 0,
            "ties": "retriever order: score descending, document ID ascending",
            "self_match": "removed after retrieval; no candidate backfill",
        },
        "quality": aggregate(scores),
        "confidence_intervals": cluster_intervals(
            scores, split["groups"], config.seed, config.bootstrap_samples
        ),
        "reference_check": reference,
        "latency": latency_summary([row["latency_ms"] for row in timings]),
        "resources": {"index_build": build.result, "measured_queries": query_resources.result},
        "measurement_protocol": {
            "device": "CPU",
            "search_threads": 1,
            "concurrent_queries": 1,
            "query_order": "sorted selected IDs for warmup; seeded shuffle per measured pass; "
            "a separate untimed full-quality pass precedes sampled timing",
            "latency_query_ids": latency_qids,
            "latency_query_selection": "all validation queries"
            if config.latency_query_limit is None
            else "seeded uniform sample of validation IDs, selected without labels; "
            "full quality evaluated separately",
            "quality_queries": len(qids),
            "cache_state": "warm in-memory index; no query-result cache; OS cache uncontrolled",
            "latency_scope": "full retrieval search call; hybrid runs both components serially "
            "at its configured depth then fuses ranks and builds its trace; "
            "dense includes query encoding, "
            "FAISS and canonical tie sorting; BM25 includes tokenization/scoring/top-k; "
            "rerank additionally includes fresh joint scoring of all top-N pairs and sorting; "
            "excludes I/O/preparation/UI/metrics",
            "preparation_scope": "hybrid builds both component indexes; BM25 indexing; "
            "rerank also loads the joint pair-scoring model; "
            "dense includes model/library loading, "
            "document encoding or verified cache loading, and FAISS construction",
            "torch_threads": getattr(
                getattr(retriever_config, "dense", retriever_config), "torch_threads", None
            ),
            "faiss_threads": getattr(
                getattr(retriever_config, "dense", retriever_config), "faiss_threads", None
            ),
            "resources_scope": "whole process in each stage, including runner and RSS sampler",
            "rss_peak": "sampled resident-memory maximum; can miss short-lived peaks",
            "cpu_percent": "one-core equivalent; sampler included; "
            "timer resolution affects short stages",
            "gpu": "not used",
            "background_load": "uncontrolled local workstation",
        },
        "environment": environment(),
        "code": code_state(),
        "source_file_sha256": source_hashes(),
    }
    if isinstance(retriever_config, RerankerConfig):
        from retrieval_lab.rerank_diagnostics import summarize

        summary["reranking_analysis"] = summarize(
            queries, retriever_config.rerank_depth, "test" if official else "validation"
        )
        summary["measurement_protocol"]["torch_threads"] = retriever_config.torch_threads
        summary["measurement_protocol"]["faiss_threads"] = (
            retriever_config.retriever.dense.faiss_threads
        )
    if official:
        summary["freeze_sha256"] = official_freeze_sha256
        summary["confidence_intervals"]["method"] = (
            "query bootstrap; singleton test-query groups; "
            "correlated claims may understate uncertainty"
        )
        summary["measurement_protocol"]["latency_query_selection"] = (
            "seeded uniform test-ID sample without labels; quality covers all official test queries"
        )
    return summary, queries, timings


def _report(summary: dict) -> str:
    quality, latency = summary["quality"], summary["latency"]
    measured = summary["resources"]["measured_queries"]
    text = (
        f"# {summary['system']} validation experiment\n\n"
        "These are measured train-derived validation results, not official SciFact test scores.\n\n"
        f"- Validation queries: {quality['query_count']}\n"
        f"- Split SHA-256: `{summary['split_sha256']}`\n"
        f"- Recall@10: {quality['recall']:.6f}\n- nDCG@10: {quality['ndcg']:.6f}\n"
        f"- Query p50 / p95: {latency['p50_ms']:.3f} / {latency['p95_ms']:.3f} ms\n"
        f"- Measured query calls: {latency['samples']}\n"
        f"- Distinct timing queries: {len(summary['measurement_protocol']['latency_query_ids'])}; "
        f"quality queries: {quality['query_count']}\n"
        "- Sampled peak process RSS during queries: "
        f"{measured['sampled_peak_rss_bytes'] / 2**20:.2f} MiB\n"
        f"- Query-stage process CPU seconds: {measured['process_cpu_seconds']:.3f}\n"
        "- Queries missing all relevant evidence: "
        f"{quality['queries_with_no_relevant_retrieved']}\n\n"
        "See manifest.json for confidence intervals, hardware, versions and measurement caveats; "
        "queries.json for rankings/missed evidence; timings.csv for raw measured calls. "
        "Unjudged documents are not proven irrelevant.\n"
    )
    if summary["run_type"] == "official_test_evaluation":
        text = (
            text.replace("validation experiment", "official held-out test experiment")
            .replace(
                "These are measured train-derived validation results, "
                "not official SciFact test scores.",
                "These are measured official SciFact test scores from settings "
                "frozen before test-label access.",
            )
            .replace("Validation queries:", "Official test queries:")
        )
    return text


def run_evaluation(
    data_dir: Path,
    split_path: Path,
    retriever_path: Path,
    evaluation_path: Path,
    output_dir: Path,
    system: str = "bm25",
    model_root: Path = Path("models"),
    cache_root: Path = Path("indexes"),
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Choose a new output directory.")
    dataset = load_dataset(data_dir, "train")
    split = load_split(split_path, dataset)
    summary, queries, timings = evaluate(
        dataset,
        split,
        load_config(retriever_path, system),
        EvaluationConfig.from_file(evaluation_path),
        model_root,
        cache_root,
    )
    summary["input_file_sha256"] = {
        "split_manifest": sha256_file(split_path),
        "retriever_config": sha256_file(retriever_path),
        "evaluation_config": sha256_file(evaluation_path),
    }
    return save_evaluation(summary, queries, timings, split, output_dir)


def save_evaluation(
    summary: dict, queries: list, timings: list, split: dict, output_dir: Path
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        (staged / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        (staged / "queries.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["repetition", "query_id", "latency_ms"])
        writer.writeheader()
        writer.writerows(timings)
        (staged / "timings.csv").write_text(stream.getvalue(), encoding="utf-8", newline="")
        (staged / "report.md").write_text(_report(summary), encoding="utf-8")
        summary["artifact_sha256"] = {
            name: sha256_file(staged / name)
            for name in ["split.json", "queries.json", "timings.csv", "report.md"]
        }
        (staged / "manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        install_directory(staged, output_dir, output_dir.parent)
    return summary


def load_report(folder: Path, dataset: Dataset) -> tuple[dict, list, list]:
    """Reject stale/mismatched artifacts before presenting them as current results."""
    summary = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    official = summary["run_type"] == "official_test_evaluation"
    if not official and summary["run_type"] != "train_derived_validation_evaluation":
        raise DatasetError("Report is not a train-derived validation run")
    if summary["dataset"]["file_sha256"] != dataset.file_sha256:
        raise DatasetError("Report dataset fingerprints differ from the current search data")
    for name in ["split.json", "queries.json", "timings.csv", "report.md"]:
        if sha256_file(folder / name) != summary["artifact_sha256"][name]:
            raise DatasetError(f"Report artifact checksum mismatch: {name}")
    if official:
        from retrieval_lab.release import test_partition

        split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        if split != test_partition(dataset, summary["freeze_sha256"]):
            raise DatasetError("Official test partition does not match the frozen report")
    else:
        split = load_split(folder / "split.json", dataset)
    if split["split_sha256"] != summary["split_sha256"]:
        raise DatasetError("Report split identity does not match its saved manifest")
    queries = json.loads((folder / "queries.json").read_text(encoding="utf-8"))
    if sorted(row["query_id"] for row in queries) != split["validation_query_ids"]:
        raise DatasetError("Report query IDs do not match the validation partition")
    if aggregate({row["query_id"]: row for row in queries}) != summary["quality"]:
        raise DatasetError("Report quality summary disagrees with its per-query evidence")
    if summary["config"]["retriever"]["system"] == "hybrid":
        from retrieval_lab.hybrid import fuse_rankings
        from retrieval_lab.types import SearchResult

        settings = summary["config"]["retriever"]
        candidates = []
        for row in queries:
            trace, qid = row["fusion_trace"], row["query_id"]
            sources = trace["source_candidate_ids"]
            if set(sources) != {"bm25", "dense"} or any(
                len(ids) > settings["candidate_depth"] for ids in sources.values()
            ):
                raise DatasetError("Hybrid candidate lists disagree with configured sources/depth")
            lists = {
                name: [SearchResult(dataset.corpus[did], 0) for did in ids]
                for name, ids in sources.items()
            }
            hits, rebuilt_trace = fuse_rankings(
                lists, settings["rank_constant"], summary["config"]["evaluation"]["cutoff"]
            )
            final = [hit for hit in hits if hit.document.doc_id != qid]
            if trace != rebuilt_trace or [(r["doc_id"], r["score"]) for r in row["results"]] != [
                (hit.document.doc_id, hit.score) for hit in final
            ]:
                raise DatasetError(
                    "Hybrid ranking/contributions disagree with saved candidate ranks"
                )
            analysis = analyze_candidates(
                trace, qid, dataset.qrels[qid], [hit.document.doc_id for hit in final]
            )
            if analysis != row["candidate_analysis"]:
                raise DatasetError("Hybrid candidate analysis disagrees with saved evidence")
            candidates.append(analysis)
        if (
            summarize_candidates(
                candidates, settings["candidate_depth"], "test" if official else "validation"
            )
            != summary["candidate_analysis"]
        ):
            raise DatasetError("Hybrid candidate summary disagrees with its per-query evidence")
    if summary["config"]["retriever"]["system"] == "rerank":
        from retrieval_lab.rerank_diagnostics import analyze, summarize, verify_trace

        cutoff = summary["config"]["evaluation"]["cutoff"]
        for row in queries:
            trace, qid = row["reranking_trace"], row["query_id"]
            # Include a self hit from the trace for structural validation; scoring filters it later.
            raw_results = [(r["doc_id"], r["raw_logit"]) for r in trace["results"]]
            verify_trace(trace, dataset, summary["config"]["retriever"], cutoff, raw_results)
            expected = [(did, score) for did, score in raw_results if did != qid]
            if [(r["doc_id"], r["score"]) for r in row["results"]] != expected:
                raise DatasetError("Reranker scored results disagree with saved trace")
            if analyze(trace, qid, dataset.qrels[qid], cutoff) != row["reranking_analysis"]:
                raise DatasetError("Reranking analysis disagrees with candidate evidence")
        if (
            summarize(
                queries,
                summary["config"]["retriever"]["rerank_depth"],
                "test" if official else "validation",
            )
            != summary["reranking_analysis"]
        ):
            raise DatasetError("Reranking summary disagrees with its per-query evidence")
    with (folder / "timings.csv").open(encoding="utf-8", newline="") as stream:
        timings = list(csv.DictReader(stream))
    if "latency_query_ids" in summary["measurement_protocol"]:
        settings = EvaluationConfig(**summary["config"]["evaluation"])
        qids = split["validation_query_ids"]
        expected_ids = (
            qids
            if settings.latency_query_limit is None
            else sorted(
                random.Random(settings.seed).sample(
                    qids, min(len(qids), settings.latency_query_limit)
                )
            )
        )
        expected_calls = {
            (rep, qid) for rep in range(1, settings.repetitions + 1) for qid in expected_ids
        }
        calls = [(int(row["repetition"]), row["query_id"]) for row in timings]
        if (
            summary["measurement_protocol"]["latency_query_ids"] != expected_ids
            or len(calls) != len(expected_calls)
            or set(calls) != expected_calls
        ):
            raise DatasetError("Latency samples do not cover the declared query IDs/repetitions")
    if latency_summary([float(row["latency_ms"]) for row in timings]) != summary["latency"]:
        raise DatasetError("Report latency summary disagrees with its recorded samples")
    return summary, queries, timings

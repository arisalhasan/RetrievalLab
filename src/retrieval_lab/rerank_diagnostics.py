"""Post-retrieval relevance analysis and integrity checks for saved reranking traces."""

from statistics import fmean

from retrieval_lab.data import Dataset, DatasetError
from retrieval_lab.hybrid import fuse_rankings
from retrieval_lab.metrics import score_query
from retrieval_lab.reranking import rerank_candidates
from retrieval_lab.types import SearchResult


def analyze(trace: dict, qid: str, qrels: dict, cutoff: int) -> dict:
    positives = {did for did, grade in qrels.items() if grade > 0}
    inputs = [row["doc_id"] for row in trace["candidates"]]
    candidate_ids = set(inputs) - {qid}
    before = [did for did in inputs[:cutoff] if did != qid]
    after = [row["doc_id"] for row in trace["results"] if row["doc_id"] != qid]
    return {
        "candidate_count": len(inputs),
        "candidate_recall": len(positives & candidate_ids) / len(positives) if positives else 0.0,
        "relevant_outside_rerank_input": sorted(positives - candidate_ids),
        "relevant_in_input_below_final_cutoff": sorted((positives & candidate_ids) - set(after)),
        "before": score_query(qid, before, qrels, cutoff),
        "after": score_query(qid, after, qrels, cutoff),
    }


def summarize(rows: list[dict], depth: int, partition: str = "validation") -> dict:
    analyses = [row["reranking_analysis"] for row in rows]
    stats = [row["reranking_trace"]["token_statistics"] for row in rows]
    return {
        "scope": f"untimed diagnostics on all {partition} queries; actual hybrid top-N input, "
        "not the larger source-list union; self IDs excluded from recall",
        "rerank_depth": depth,
        "query_count": len(rows),
        "candidate_recall": fmean(row["candidate_recall"] for row in analyses),
        "mean_candidates": fmean(row["candidate_count"] for row in analyses),
        "pre_rerank_recall_at_10": fmean(row["before"]["recall"] for row in analyses),
        "pre_rerank_ndcg_at_10": fmean(row["before"]["ndcg"] for row in analyses),
        "queries_with_evidence_outside_rerank_input": sum(
            bool(row["relevant_outside_rerank_input"]) for row in analyses
        ),
        "queries_with_input_evidence_lost_below_cutoff": sum(
            bool(row["relevant_in_input_below_final_cutoff"]) for row in analyses
        ),
        "ndcg_outcomes": {
            "higher": sum(row["after"]["ndcg"] > row["before"]["ndcg"] + 1e-12 for row in analyses),
            "lower": sum(row["after"]["ndcg"] < row["before"]["ndcg"] - 1e-12 for row in analyses),
            "equal": sum(
                abs(row["after"]["ndcg"] - row["before"]["ndcg"]) <= 1e-12 for row in analyses
            ),
        },
        "input_token_statistics": {
            "pairs": sum(row["pairs"] for row in stats),
            "pairs_truncated": sum(row["pairs_truncated"] for row in stats),
            "max_joint_input_tokens": max(row.get("max_joint_input_tokens", 0) for row in stats),
        },
    }


def verify_trace(trace: dict, dataset: Dataset, config: dict, cutoff: int, results: list) -> None:
    sources = trace["fusion_trace"]["source_candidate_ids"]
    if set(sources) != {"bm25", "dense"} or any(
        len(ids) > config["retriever"]["candidate_depth"] for ids in sources.values()
    ):
        raise DatasetError("Reranker source candidates do not match configured sources/depth")
    if any(did not in dataset.corpus for ids in sources.values() for did in ids):
        raise DatasetError("Unknown document in reranker source candidates")
    lists = {
        name: [SearchResult(dataset.corpus[did], 0) for did in ids] for name, ids in sources.items()
    }
    candidates, fusion = fuse_rankings(
        lists, config["retriever"]["rank_constant"], config["rerank_depth"]
    )
    if trace["fusion_trace"] != fusion or len(trace["candidates"]) != len(candidates):
        raise DatasetError("Rerank input disagrees with saved fusion candidate order")
    scores = [row["reranker_score"] for row in trace["candidates"]]
    _, rebuilt = rerank_candidates(candidates, scores, cutoff)
    if any(trace[name] != rebuilt[name] for name in ["candidates", "results"]):
        raise DatasetError("Reranker ranking/scores disagree with candidate trace")
    if trace["rerank_depth"] != config["rerank_depth"]:
        raise DatasetError("Saved rerank depth differs from config")
    # The caller applies BEIR self filtering after cutoff when comparing scored results.
    expected = [(row["doc_id"], row["raw_logit"]) for row in rebuilt["results"]]
    if results != expected:
        raise DatasetError("Scored reranker results disagree with saved logits")

"""Run from the repository root: python -m streamlit run app.py."""

import json
import os
from pathlib import Path
from time import perf_counter

import streamlit as st

from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.dashboard import render_comparison, render_validation
from retrieval_lab.data import DatasetError, load_dataset, sha256_file

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RetrievalLab · SciFact search", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading SciFact and building the BM25 index…", max_entries=2)
def load_search(folder: str, fingerprint: tuple[str, ...], config_json: str):
    """Content hashes invalidate the cache when data changes, even at the same path."""
    dataset = load_dataset(folder, split="train")
    values = json.loads(config_json)
    values["document_fields"] = tuple(values["document_fields"])
    config = BM25Config(**values)
    started = perf_counter()
    index = BM25Index(list(dataset.corpus.values()), config)
    return dataset, index, (perf_counter() - started) * 1000


@st.cache_resource(show_spinner="Loading the local dense search index…", max_entries=2)
def load_dense_search(
    folder: str,
    fingerprint: tuple[str, ...],
    config_json: str,
    model_root: str,
    cache_root: str,
    model_manifest_hash: str,
):
    from retrieval_lab.dense import build_dense
    from retrieval_lab.dense_config import DenseConfig

    dataset = load_dataset(folder, split="train")
    values = json.loads(config_json)
    values["document_fields"] = tuple(values["document_fields"])
    config = DenseConfig(**values)
    started = perf_counter()
    index = build_dense(
        list(dataset.corpus.values()), config, Path(model_root), Path(cache_root), allow_build=False
    )
    return dataset, index, (perf_counter() - started) * 1000


@st.cache_resource(show_spinner="Loading the local cross-encoder…", max_entries=2)
def load_pair_scorer(config_json: str, model_root: str, manifest_hash: str):
    from retrieval_lab.cross_encoder import CrossEncoderScorer
    from retrieval_lab.reranking import RerankerConfig

    started = perf_counter()
    scorer = CrossEncoderScorer(RerankerConfig.from_dict(json.loads(config_json)), Path(model_root))
    return scorer, (perf_counter() - started) * 1000


def main() -> None:
    st.title("RetrievalLab")
    st.write("Retrieve scientific evidence, then inspect how a neural reranker changes the order.")
    with st.sidebar:
        st.header("Search setup")
        st.write("**BEIR SciFact**")
        st.caption("Search, compare, and audit reproducible experiments")
        source = st.radio("Query source", ["Your query", "Training claim"])
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=10)
        methods = ["BM25", "Dense", "Hybrid", "Reranked"]
        if (ROOT / "configs/trained-reranker.json").exists():
            methods.append("Fine-tuned")
        method = st.radio("Search method", methods)
        st.divider()
        st.caption(
            "Training claims are available for exploration. "
            "Official test scores are reported separately after settings are frozen."
        )
    folder = Path(os.environ.get("RETRIEVALLAB_DATA_DIR", ROOT / "data/scifact"))
    try:
        config = BM25Config.from_file(ROOT / "configs/bm25.json")
        fingerprint = tuple(
            sha256_file(folder / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv")
        )
        config_json = json.dumps(config.to_dict(), sort_keys=True)
        dataset, index, build_ms = load_search(str(folder.resolve()), fingerprint, config_json)
        if method in {"Dense", "Hybrid", "Reranked", "Fine-tuned"}:
            from retrieval_lab.dense_config import DenseConfig
            from retrieval_lab.hybrid import HybridConfig, HybridIndex
            from retrieval_lab.model_store import verify_model
            from retrieval_lab.reranking import RerankedIndex, RerankerConfig
            from retrieval_lab.splits import canonical_hash

            rerank_config = (
                RerankerConfig.from_file(
                    ROOT
                    / "configs"
                    / ("trained-reranker.json" if method == "Fine-tuned" else "reranker.json")
                )
                if method in {"Reranked", "Fine-tuned"}
                else None
            )
            hybrid_config = (
                rerank_config.retriever
                if rerank_config
                else HybridConfig.from_file(ROOT / "configs/hybrid.json")
                if method == "Hybrid"
                else None
            )
            config = (
                hybrid_config.dense
                if hybrid_config
                else DenseConfig.from_file(ROOT / "configs/dense.json")
            )
            model_root = Path(os.environ.get("RETRIEVALLAB_MODEL_ROOT", ROOT / "models"))
            cache_root = Path(os.environ.get("RETRIEVALLAB_CACHE_ROOT", ROOT / "indexes"))
            manifest = verify_model(config, model_root)
            config_json = json.dumps(config.to_dict(), sort_keys=True)
            dataset, index, build_ms = load_dense_search(
                str(folder.resolve()),
                fingerprint,
                config_json,
                str(model_root),
                str(cache_root),
                canonical_hash(manifest),
            )
            if hybrid_config:
                _, lexical, lexical_ms = load_search(
                    str(folder.resolve()),
                    fingerprint,
                    json.dumps(hybrid_config.bm25.to_dict(), sort_keys=True),
                )
                index = HybridIndex(lexical, index, hybrid_config)
                config = hybrid_config
                config_json = json.dumps(config.to_dict(), sort_keys=True)
                build_ms += lexical_ms
            if rerank_config:
                from retrieval_lab.reranker_store import verify_model as verify_reranker

                reranker_manifest = verify_reranker(rerank_config, model_root)
                config = rerank_config
                config_json = json.dumps(config.to_dict(), sort_keys=True)
                scorer, model_ms = load_pair_scorer(
                    config_json, str(model_root), canonical_hash(reranker_manifest)
                )
                index = RerankedIndex(index, scorer, config)
                build_ms += model_ms
    except (DatasetError, OSError, ValueError, TypeError, ImportError) as exc:
        st.error("The SciFact search index could not be loaded.")
        st.write("From the project folder, prepare the required local files and reload:")
        commands = (
            "python -m retrieval_lab.cli download"
            if method == "BM25"
            else (
                "python -m retrieval_lab.cli model-download\n"
                "python -m retrieval_lab.cli build-dense"
            )
        )
        if method in {"Reranked", "Fine-tuned"}:
            commands += "\npython -m retrieval_lab.cli reranker-download"
        st.code(commands, language="bash")
        if method in {"Dense", "Hybrid", "Reranked", "Fine-tuned"}:
            st.caption(
                "Dense and hybrid search need the optional CPU dependencies in the setup guide."
            )
        with st.expander("Show details"):
            st.text(str(exc))
        st.stop()

    with st.sidebar:
        st.write(f"{len(dataset.corpus):,} searchable abstracts")
        st.write(f"{len(dataset.queries):,} training claims")
        st.caption(
            f"k1 = {config.k1} · b = {config.b}"
            if method == "BM25"
            else ("MiniLM · 384 dimensions · cosine similarity · CPU")
            if method == "Dense"
            else f"RRF constant {config.rank_constant} · "
            f"up to {config.candidate_depth} candidates per method"
            if method == "Hybrid"
            else f"TinyBERT · top {config.rerank_depth} hybrid candidates · "
            f"{config.max_length} joint tokens"
        )

    search_tab, validation_tab, comparison_tab, method_tab, final_tab = st.tabs(
        [
            "Search evidence",
            "Validation & failures",
            "Compare systems",
            "Method & reproducibility",
            "Final test",
        ]
    )
    with search_tab:
        query_id = None
        with st.form("search_form"):
            if source == "Training claim":
                query_id = st.selectbox(
                    "Choose a training claim",
                    list(dataset.queries),
                    format_func=lambda qid: f"{qid} · {dataset.queries[qid]}",
                )
                query = dataset.queries[query_id]
                st.write(query)
            else:
                query = st.text_input(
                    "Search titles and abstracts",
                    placeholder="Try: cell division and mitosis",
                    max_chars=2000,
                )
            submitted = st.form_submit_button("Search", type="primary")
        signature = (5, query, query_id, top_k, fingerprint, config_json)
        if submitted:
            if not query.strip():
                st.warning("Enter a query before searching.")
            else:
                started = perf_counter()
                trace = None
                if method in {"Hybrid", "Reranked", "Fine-tuned"}:
                    hits, trace = index.search_with_trace(query, top_k)
                else:
                    hits = index.search(query, top_k)
                elapsed_ms = (perf_counter() - started) * 1000
                st.session_state["search"] = (signature, hits, elapsed_ms, trace)
        saved = st.session_state.get("search")
        if saved is None or saved[0] != signature:
            st.info("Choose a training claim or enter your own query, then select Search.")
        else:
            _, hits, elapsed_ms, trace = saved
            st.divider()
            st.subheader(f"{len(hits)} matching abstracts")
            st.caption(
                f"Query time: {elapsed_ms:.2f} ms · Full retrieval call "
                "(including query encoding for dense/hybrid, "
                "both methods plus fusion for hybrid, and fresh pair scoring for reranked); "
                "excludes data loading, index construction, and rendering."
            )
            if not hits:
                st.info(
                    "No shared words found. Try a shorter query or a different scientific term."
                    if method == "BM25"
                    else "No results. Enter a query containing words or numbers."
                )
            for rank, hit in enumerate(hits, 1):
                doc = hit.document
                st.subheader(f"{rank}. {doc.title or 'Untitled abstract'}")
                score_label = {
                    "BM25": "BM25 score",
                    "Dense": "Cosine similarity",
                    "Hybrid": "RRF score",
                    "Reranked": "Cross-encoder logit",
                    "Fine-tuned": "Fine-tuned cross-encoder logit",
                }[method]
                st.caption(f"Document {doc.doc_id} · {score_label} {hit.score:.6f}")
                if trace and method == "Hybrid":
                    explanation = next(
                        row for row in trace["results"] if row["doc_id"] == doc.doc_id
                    )
                    with st.expander(f"Why this rank? · {doc.doc_id}"):
                        for name in ["bm25", "dense"]:
                            part = explanation["sources"].get(name)
                            st.write(
                                f"{name.upper()}: rank {part['rank']}; contributes "
                                f"1 / ({config.rank_constant} + {part['rank']}) "
                                f"= {part['contribution']:.6f}"
                                if part
                                else f"{name.upper()}: outside its candidate list; contributes 0."
                            )
                        st.caption(
                            "Add the contributions to get the RRF score. "
                            "Raw search scores are not added."
                        )
                if trace and method in {"Reranked", "Fine-tuned"}:
                    movement = next(row for row in trace["results"] if row["doc_id"] == doc.doc_id)
                    with st.expander(f"How the rank changed · {doc.doc_id}"):
                        st.write(
                            f"Hybrid candidate rank {movement['retrieval_rank']} → "
                            f"reranked rank {movement['reranked_rank']}."
                        )
                        st.caption(
                            "The model reads the query and abstract together. This raw logit is "
                            "a relative relevance score, not a calibrated probability "
                            "or truth assessment."
                        )
                if query_id is not None:
                    relevance = dataset.qrels[query_id].get(doc.doc_id)
                    st.write(
                        "**Judged relevant**"
                        if relevance and relevance > 0
                        else "Unjudged / no positive judgment for this claim"
                    )
                st.write(doc.text[:450] + ("…" if len(doc.text) > 450 else ""))
                with st.expander(f"Read full abstract · {doc.doc_id}"):
                    st.write(doc.text)
                st.divider()
            payload = {
                "run_type": "interactive_search_not_benchmark",
                "query": query,
                "training_query_id": query_id,
                "split": "train",
                "top_k": top_k,
                "config": config.to_dict(),
                "file_sha256": dataset.file_sha256,
                "query_time_ms": elapsed_ms,
                "fusion_trace": trace.get("fusion_trace")
                if trace and method in {"Reranked", "Fine-tuned"}
                else trace,
                "reranking_trace": trace if method in {"Reranked", "Fine-tuned"} else None,
                "results": [
                    {"rank": n, "doc_id": hit.document.doc_id, "score": hit.score}
                    for n, hit in enumerate(hits, 1)
                ],
            }
            st.download_button(
                "Download this search",
                json.dumps(payload, indent=2),
                file_name="retrievallab-search.json",
                mime="application/json",
            )
        st.caption(
            "Search scores measure retrieval similarity, not scientific truth or probability. "
            "A relevant abstract may support or refute a claim; unjudged does not mean irrelevant."
        )
    with validation_tab:
        render_validation(
            dataset, folder, Path(os.environ.get("RETRIEVALLAB_RUNS_DIR", ROOT / "runs"))
        )
    with comparison_tab:
        render_comparison(
            dataset, folder, Path(os.environ.get("RETRIEVALLAB_RUNS_DIR", ROOT / "runs"))
        )
    with method_tab:
        st.header("A baseline you can inspect")
        st.write(
            "BM25 rewards matching words that are rare across the collection. Repeating a word "
            "helps less each time, and document length is normalized so long abstracts do not "
            "win just because they contain more words."
        )
        st.write(
            "Titles and abstracts are combined, lowercased with Unicode case folding, and split "
            "into letter/digit tokens. This version uses no stemming or stopword removal."
        )
        st.write(
            "The validation tab reads saved Recall@10, nDCG@10, latency and process-resource "
            "measurements. Run experiments with the CLI so the dashboard does not affect timing. "
            "Dense search encodes text with a pinned MiniLM model and uses exact FAISS cosine "
            "search. Document vectors are cached; every search still encodes its query. "
            "Long inputs are truncated to 256 model tokens. "
            "The default hybrid search takes up to 100 candidates "
            "from each method and adds 1 / (60 + rank) per appearance. It uses positions rather "
            "than mixing the different score scales. Open Why this rank? to inspect contributions."
        )
        st.write(
            "Reranked search reads each query and one of the top 50 hybrid candidates jointly "
            "with a pinned two-layer TinyBERT model. It can reorder only that candidate set. "
            "Inputs share a 512-token budget. The Fine-tuned method uses a separately trained, "
            "verified checkpoint. Training runs explicitly from the command line."
        )
        st.caption(
            f"Index construction on this process: {build_ms:.1f} ms. Not a benchmark average."
        )
        with st.expander("Inspect configuration and dataset fingerprints"):
            st.json(
                {
                    "config": config.to_dict(),
                    "split": dataset.split,
                    "file_sha256": dataset.file_sha256,
                }
            )
    with final_tab:
        from retrieval_lab.release_dashboard import render_final

        render_final(ROOT, folder)


if __name__ == "__main__":
    main()

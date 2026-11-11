from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def start_app(monkeypatch, folder):
    monkeypatch.setenv("RETRIEVALLAB_DATA_DIR", str(folder))
    monkeypatch.setenv("RETRIEVALLAB_RUNS_DIR", str(folder / "no-runs"))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    return app


def test_search_flow_and_no_results(monkeypatch, mini_dataset):
    app = start_app(monkeypatch, mini_dataset)
    app.text_input[0].set_value("planets")
    app.button[0].click().run()
    assert not app.exception
    assert any("Space science" in h.value for h in app.subheader)
    assert any("BM25 score" in c.value for c in app.caption)
    app.text_input[0].set_value("qzxqzxunknown")
    app.button[0].click().run()
    assert any("No shared words" in msg.value for msg in app.info)
    assert not app.exception


def test_training_claims_show_judgments_without_test_queries(monkeypatch, mini_dataset):
    app = start_app(monkeypatch, mini_dataset)
    app.radio[0].set_value("Training claim").run()
    assert app.selectbox[0].options == ["train-1 · cell mitosis"]
    app.button[0].click().run()
    assert not app.exception
    assert any("Judged relevant" in item.value for item in app.markdown)
    assert any("Unjudged" in item.value for item in app.markdown)


def test_missing_dataset_has_recovery_instructions(monkeypatch, tmp_path):
    app = start_app(monkeypatch, tmp_path / "absent")
    assert app.error
    assert "download" in app.code[0].value


@pytest.mark.parametrize("method", ["Dense", "Hybrid", "Reranked"])
def test_missing_dense_model_explains_preparation(monkeypatch, mini_dataset, tmp_path, method):
    monkeypatch.setenv("RETRIEVALLAB_MODEL_ROOT", str(tmp_path / "missing-model"))
    app = start_app(monkeypatch, mini_dataset)
    app.radio[1].set_value(method).run()
    assert not app.exception
    assert app.error
    assert "model-download" in app.code[0].value
    assert "build-dense" in app.code[0].value


def test_empty_query_and_changing_query_do_not_show_stale_results(monkeypatch, mini_dataset):
    app = start_app(monkeypatch, mini_dataset)
    app.button[0].click().run()
    assert app.warning
    app.text_input[0].set_value("planets")
    app.button[0].click().run()
    assert any("Space science" in h.value for h in app.subheader)
    app.text_input[0].set_value("mitosis").run()
    assert not any("Space science" in h.value for h in app.subheader)
    assert not app.exception


def test_hybrid_search_displays_live_rank_contributions(monkeypatch, mini_dataset, tmp_path):
    pytest.importorskip("faiss")
    from test_dense import SyntheticEncoder

    from retrieval_lab import encoders
    from retrieval_lab.data import load_dataset
    from retrieval_lab.dense import prepare_embeddings
    from retrieval_lab.hybrid import HybridConfig

    config = HybridConfig()
    cache = tmp_path / "cache"
    dataset = load_dataset(mini_dataset)
    prepare_embeddings(list(dataset.corpus.values()), config.dense, SyntheticEncoder(), cache)
    monkeypatch.setenv("RETRIEVALLAB_CACHE_ROOT", str(cache))
    # Import the real encoder before replacing the app's model-store boundary, so its
    # module-level verification alias never captures the temporary test stub.
    monkeypatch.setattr(encoders, "SentenceEncoder", lambda *args: SyntheticEncoder())
    monkeypatch.setattr("retrieval_lab.model_store.verify_model", lambda *args: {"synthetic": True})
    app = start_app(monkeypatch, mini_dataset)
    app.radio[1].set_value("Hybrid").run()
    app.text_input[0].set_value("planets")
    app.button[0].click().run()
    assert not app.exception
    assert any("Space science" in item.value for item in app.subheader)
    assert any("RRF score" in item.value for item in app.caption)
    assert any("BM25: rank 1" in item.value for item in app.markdown)
    assert any("DENSE: rank 1" in item.value for item in app.markdown)
    trace = app.session_state["search"][3]
    assert trace["results"][0]["doc_id"] == "d2"
    assert trace["results"][0]["rrf_score"] == 2 / 61
    app.radio[1].set_value("BM25").run()
    assert not app.exception
    assert not any("RRF score" in item.value for item in app.caption)


def test_reranked_search_shows_scores_and_rank_changes(monkeypatch, mini_dataset, tmp_path):
    pytest.importorskip("faiss")
    from test_dense import SyntheticEncoder
    from test_reranking import SyntheticPairScorer

    from retrieval_lab import cross_encoder, encoders
    from retrieval_lab.data import load_dataset
    from retrieval_lab.dense import prepare_embeddings
    from retrieval_lab.reranking import RerankerConfig

    config = RerankerConfig()
    cache = tmp_path / "cache"
    prepare_embeddings(
        list(load_dataset(mini_dataset).corpus.values()),
        config.retriever.dense,
        SyntheticEncoder(),
        cache,
    )
    monkeypatch.setenv("RETRIEVALLAB_CACHE_ROOT", str(cache))
    # Import real neural modules first, then replace only external model/file boundaries.
    monkeypatch.setattr(encoders, "SentenceEncoder", lambda *args: SyntheticEncoder())
    monkeypatch.setattr(cross_encoder, "CrossEncoderScorer", lambda *args: SyntheticPairScorer())
    monkeypatch.setattr("retrieval_lab.model_store.verify_model", lambda *args: {"synthetic": True})
    monkeypatch.setattr(
        "retrieval_lab.reranker_store.verify_model", lambda *args: {"synthetic": True}
    )
    app = start_app(monkeypatch, mini_dataset)
    app.radio[1].set_value("Reranked").run()
    app.text_input[0].set_value("planets")
    app.button[0].click().run()
    assert not app.exception
    assert any("Space science" in item.value for item in app.subheader)
    assert any("Cross-encoder logit" in item.value for item in app.caption)
    assert any("Hybrid candidate rank" in item.value for item in app.markdown)
    trace = app.session_state["search"][3]
    assert trace["results"][0]["doc_id"] == "d2"
    app.radio[1].set_value("BM25").run()
    assert not app.exception and not any("Cross-encoder logit" in c.value for c in app.caption)

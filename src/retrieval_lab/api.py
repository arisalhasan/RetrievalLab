"""Local HTTP interface over the same search engines used by experiments."""

import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrieval_lab import __version__
from retrieval_lab.data import load_dataset
from retrieval_lab.retrievers import build_retriever, load_config

System = Literal["bm25", "dense", "hybrid", "rerank", "trained"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    system: System = "bm25"
    top_k: int = Field(default=10, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Query must contain text")
        return value.strip()


class SearchHit(BaseModel):
    rank: int
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    run_type: str = "interactive_search_not_benchmark"
    query: str
    system: str
    results: list[SearchHit]
    query_time_ms: float
    config: dict
    dataset_sha256: dict
    trace: dict | None


class SearchService:
    """Lazy indexes; bounded, serial CPU inference. Never downloads model assets."""

    def __init__(self, root: Path = Path("."), data_dir: Path | None = None):
        self.root = root
        self.data_dir = data_dir or root / "data/scifact"
        self.dataset = None
        self.indexes = {}
        self.gate = threading.Lock()

    def search(self, request: SearchRequest):
        if not self.gate.acquire(blocking=False):
            raise HTTPException(
                429, "Search engine busy; retry shortly", headers={"Retry-After": "1"}
            )
        try:
            if self.dataset is None:
                self.dataset = load_dataset(self.data_dir, "train")
            if request.system not in self.indexes:
                names = {"rerank": "reranker", "trained": "trained-reranker"}
                path = self.root / "configs" / f"{names.get(request.system, request.system)}.json"
                config = load_config(
                    path, "rerank" if request.system == "trained" else request.system
                )
                # Preparation is explicit in the API timing and excluded from query_time_ms.
                index = build_retriever(
                    list(self.dataset.corpus.values()),
                    config,
                    self.root / "models",
                    self.root / "indexes",
                    allow_build=False,
                )
                self.indexes[request.system] = (config, index)
            config, index = self.indexes[request.system]
            if request.top_k > getattr(
                config, "rerank_depth", getattr(config, "candidate_depth", 20)
            ):
                raise HTTPException(422, "top_k exceeds this system's candidate depth")
            started = perf_counter()
            if hasattr(index, "search_with_trace"):
                hits, trace = index.search_with_trace(request.query, request.top_k)
            else:
                hits, trace = index.search(request.query, request.top_k), None
            elapsed = (perf_counter() - started) * 1000
            return SearchResponse(
                query=request.query,
                system=request.system,
                results=[
                    SearchHit(
                        rank=n,
                        doc_id=h.document.doc_id,
                        title=h.document.title,
                        text=h.document.text,
                        score=h.score,
                    )
                    for n, h in enumerate(hits, 1)
                ],
                query_time_ms=elapsed,
                config=config.to_dict(),
                dataset_sha256=self.dataset.file_sha256,
                trace=trace,
            )
        finally:
            self.gate.release()


def create_app(service: SearchService | None = None) -> FastAPI:
    service = service or SearchService()
    app = FastAPI(
        title="RetrievalLab",
        version=__version__,
        description="Local scientific-evidence retrieval. Scores are not truth probabilities.",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "loaded_systems": sorted(service.indexes)}

    @app.get("/systems")
    def systems():
        return {
            "systems": list(System.__args__),
            "default": "bm25",
            "preparation": "Lazy local initialization; neural assets must be installed first",
            "timing": "Query only, excludes first-request initialization and HTTP overhead",
        }

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        try:
            return service.search(request)
        except (OSError, ValueError, ImportError) as exc:
            logging.getLogger(__name__).warning("Search unavailable: %s", exc)
            raise HTTPException(
                503, "Search assets unavailable; see local server log and setup guide"
            ) from exc

    return app


app = create_app()

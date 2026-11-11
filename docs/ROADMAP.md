# Milestones and repository plan

Stop at each milestone for a student verification and explanation checkpoint.

| Milestone | Deliverable | Completion evidence |
| --- | --- | --- |
| 1 · BM25 | Official loader, local BM25, Streamlit, recorded CLI search, tests and docs | Verified data, working search, actual test report |
| 2 · Evaluation (implemented) | Frozen grouped train/validation IDs; Recall@10, nDCG@10, failures, latency and process CPU/RAM | Hand metric tests, agreement with pytrec_eval (BEIR's metric backend), actual validation artifacts; full BEIR API not run |
| 3 · Dense (implemented) | Version-pinned Sentence Transformers/PyTorch encoder, FAISS exact search, paired dashboard | Embedding metadata, independent exact-search checks, same-split comparison |
| 4 · Hybrid (implemented) | Reciprocal rank fusion, rank explanations and candidate diagnostics | RRF hand/rational checks; frozen defaults and declared depth ablation; paired validation runs |
| 5 · Reranking (implemented) | Pinned TinyBERT over top-50 hybrid candidates; rank movement and input diagnostics | Real-logit oracle, full validation quality, declared shared timing sample, paired comparisons |
| 6 · Platform (implemented) | FastAPI; shared search core; CSV/JSON/PNG/SVG comparison exports | HTTP integration tests, same model configurations, real local smoke requests |
| 7 · Training experiment (implemented) | Two-epoch CPU TinyBERT fine-tuning; parent eligible in checkpoint selection | Compute plan explained first; 647 training / 162 validation IDs, saved pairs and safe hashed checkpoints |
| 8 · Held-out release (implemented) | Pre-test freeze, explicit official test, complete reports and final dashboard | Tamper/drift guards, saved official-test artifacts, uncertainty, failures and execution limits |

Retrieval finds candidate evidence. Reranking improves order only within that candidate set.

## Modules and next additions

```text
src/retrieval_lab/
  splits.py                    Implemented: grouped train/validation manifests
  metrics.py                   Implemented: Recall/nDCG and reference checks
  evaluation.py                Implemented: shared BM25/dense experiment runner
  profiling.py                 Implemented: latency and process resources
  dashboard.py                 Implemented: report/failure-analysis viewer
  provenance.py                Implemented: hardware, versions, code hashes
  dense_config.py              Implemented: pinned model/settings
  model_store.py               Implemented: checked public model download
  encoders.py                  Implemented: local Sentence Transformers/PyTorch
  dense.py                     Implemented: document cache and query encoding
  vectors.py                   Implemented: FAISS exact cosine + NumPy oracle
  comparison.py                Implemented: paired validation comparisons
  hybrid.py                    Implemented: configured RRF and label-free contribution traces
  reranking.py                 Implemented: bounded candidate reranking/config/trace
  cross_encoder.py             Implemented: local pretrained pair scoring
  reranker_store.py            Implemented: verified TinyBERT assets
  rerank_diagnostics.py         Implemented: input ceilings, losses and report verification
  training.py                  Reproducible pairs, CPU training and checkpoint verification
  api.py                       Validated, serialized local FastAPI search
  exports.py                   pandas CSV and matplotlib PNG/SVG figures
  release.py                   Immutable pre-test freeze and auditable final evaluation
  final_report.py              Held-out tables, paired intervals, figures and failure analysis
  release_dashboard.py         Verified held-out results and missed-evidence browser
configs/                       Per-experiment configurations
scripts/                       Repeatable batch/reporting commands
runs/                          Rankings, metrics, environment, reports
tests/                         Unit, integration, API and UI checks
```

The completed platform adds optional FastAPI, Uvicorn, HTTPX tests, and matplotlib exports. The full BEIR neural API is not required or executed: official BEIR files and its metric backend, pytrec_eval, provide the benchmark interface and reference checks. No service, database, GPU, paid API, or unnecessary infrastructure is added.

## Optional fine-tuning compute preview

The user authorized stages 6–8 together. Before implementation, the CPU plan was explained: two epochs of the approximately 4.4M-parameter TinyBERT, batch 8, float32, 512 tokens, one CPU thread; several minutes to tens of minutes expected. This avoids an unmeasured GPU estimate. Actual memory/time and epoch selection appear in the recorded training result. Read [milestone 7](MILESTONE_7.md) before rerunning it. Student verification checkpoints and three interview questions remain in each milestone guide.

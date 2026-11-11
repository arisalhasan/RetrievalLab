# RetrievalLab — official SciFact test, release v1

Measured on 2026-11-11T12:55:09.066203+00:00. All 300 official test queries were evaluated.

**Selected before test: hybrid-10.** The selection remains unchanged after test results.
Freeze SHA-256: `9431036e9c0a3b866a89b21cad115b2605f8f101f6881619c615c62828650642`.

| System | Recall@10 | nDCG@10 | p50 / p95 ms | Query peak RSS MiB | Complete misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.790944 | 0.661676 | 20.05 / 39.77 | 98.00 | 57 |
| dense | 0.783333 | 0.645082 | 36.69 / 46.31 | 506.34 | 62 |
| hybrid-100 | 0.810556 | 0.683420 | 59.11 / 89.41 | 558.02 | 52 |
| hybrid-10 | 0.844333 | 0.701476 | 60.87 / 100.31 | 558.38 | 43 |
| pretrained-reranker | 0.809278 | 0.679441 | 1305.72 / 1514.79 | 864.14 | 54 |
| trained-reranker | 0.827722 | 0.697833 | 1265.68 / 1384.56 | 879.41 | 48 |

## Uncertainty and paired comparisons

95% query-bootstrap intervals use 1,000 seed-42 samples. Related claims can make these intervals too narrow. Differences are exploratory, without multiple-comparison adjustment. A point-score advantage is not automatically a reliable improvement.

- bm25: Recall [0.7462, 0.8325], nDCG [0.6152, 0.7023].
- dense: Recall [0.7353, 0.8263], nDCG [0.5957, 0.6890].
- hybrid-100: Recall [0.7644, 0.8517], nDCG [0.6377, 0.7251].
- hybrid-10: Recall [0.8027, 0.8833], nDCG [0.6568, 0.7418].
- pretrained-reranker: Recall [0.7613, 0.8531], nDCG [0.6341, 0.7239].
- trained-reranker: Recall [0.7827, 0.8697], nDCG [0.6516, 0.7390].
- dense minus bm25: nDCG -0.0166, paired interval [-0.0523, +0.0230]; Recall -0.0076, interval [-0.0521, +0.0382].
- hybrid-100 minus bm25: nDCG +0.0217, paired interval [-0.0016, +0.0438]; Recall +0.0196, interval [-0.0083, +0.0466].
- hybrid-10 minus bm25: nDCG +0.0398, paired interval [+0.0189, +0.0622]; Recall +0.0534, interval [+0.0254, +0.0831].
- pretrained-reranker minus bm25: nDCG +0.0178, paired interval [-0.0147, +0.0501]; Recall +0.0183, interval [-0.0209, +0.0569].
- trained-reranker minus bm25: nDCG +0.0362, paired interval [+0.0030, +0.0685]; Recall +0.0368, interval [-0.0012, +0.0763].

## Failure analysis of the pre-test selected system

Examples below are selected after testing for explanation, never for further model selection. Unjudged documents may also be relevant; this is analysis against the available official labels.

### Claim 1049: Ribosomopathies have a low degree of cell and tissue specific pathology.

Recall@10 0.0000; nDCG@10 0.0000. Missed official relevant IDs: 12486491.

- 12486491: Ribosome-Mediated Specificity in Hox mRNA Translation and Vertebrate Tissue Patterning

Candidate analysis: `{"union_size_after_self_filter": 18, "union_recall": 0.0, "source_recall": {"bm25": 0.0, "dense": 0.0}, "relevant_in_candidates": [], "relevant_missing_from_candidates": ["12486491"], "relevant_dropped_by_fusion": []}`.

### Claim 1110: Suboptimal nutrition is not predictive of chronic disease

Recall@10 0.0000; nDCG@10 0.0000. Missed official relevant IDs: 13770184.

- 13770184: Global, regional, and national comparative risk assessment of 79 behavioural, environmental and occupational, and metabolic risks or clusters of risks, 1990–2015: a systematic analysis for the Global Burden of Disease Study 2015

Candidate analysis: `{"union_size_after_self_filter": 20, "union_recall": 0.0, "source_recall": {"bm25": 0.0, "dense": 0.0}, "relevant_in_candidates": [], "relevant_missing_from_candidates": ["13770184"], "relevant_dropped_by_fusion": []}`.

### Claim 1175: The PPR MDA5 has two N-terminal CARD domains.

Recall@10 0.0000; nDCG@10 0.0000. Missed official relevant IDs: 31272411.

- 31272411: Immune signaling by RIG-I-like receptors.

Candidate analysis: `{"union_size_after_self_filter": 19, "union_recall": 0.0, "source_recall": {"bm25": 0.0, "dense": 0.0}, "relevant_in_candidates": [], "relevant_missing_from_candidates": ["31272411"], "relevant_dropped_by_fusion": []}`.

## Performance and reproducibility limits

Every method uses the same 16 label-independent timing queries, one warmup and three measured passes (48 calls), with fresh inference. Quality uses the complete test. The CPU workstation and background load are uncontrolled; timings are descriptive. Sampled RSS covers the whole process, not isolated model weights. Preparation is separate. No GPU, other-OS, public deployment, load test, or clean-machine installation was executed. The full BEIR neural API was not executed; official BEIR files and pytrec_eval were used.

All system manifests record source/model/config/data hashes, seeds, runtime versions, raw rankings, timing samples, metric definitions, and independent reference checks. Git commit identity is unavailable in this mirrored workspace. Historical development runs remain separate and are not presented as test results.

## Portfolio description

Built RetrievalLab, a reproducible scientific-evidence search platform comparing BM25, dense FAISS retrieval, reciprocal rank fusion, and pretrained/fine-tuned neural reranking. Implemented a shared FastAPI/Streamlit core, leakage-resistant development splits, versioned experiment evidence, held-out evaluation, and latency/memory diagnostics. Reported measured tradeoffs and regressions instead of assuming neural models always improve ranking.

# BM25: first measured validation baseline

Measured on 7 November 2026 with RetrievalLab 0.2.0. These are **train-derived development-validation results**, not official SciFact test scores, leaderboard numbers, or claims about other retrieval systems.

## Dataset and frozen configuration

- Official BEIR SciFact corpus: 5,183 documents. Source judgments: official train only (809 queries).
- Derived partitions: 647 train / 162 validation, using 490 related-claim groups (98 validation groups; largest group 8). Shared positive-evidence overlap between partitions: zero.
- Split seed: 42. Target validation fraction: 0.2. Grouping: shared positive evidence or token-set Jaccard ≥0.8, then connected components.
- Split SHA-256: `091e0d820c4dac0fd0ea80a133773da72984b7890ef15a66b053c8338d8b34c6`.
- BM25 implementation 1.0.0; k1=1.2, b=0.75; positive IDF; title plus abstract; Unicode word/casefold tokenizer; unique query terms; no stemming/stopword removal. No parameter tuning was performed for this baseline.
- Hardware: Windows 11 x86-64; AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD; 8 physical / 16 logical CPUs; Python 3.12.14. Full packages/hardware are in each run manifest.

## Actual quality results

| Measure | First run | Reproduction |
| --- | ---: | ---: |
| Query macro Recall@10 | 0.8013374486 | 0.8013374486 |
| Query macro nDCG@10 | 0.6673015967 | 0.6673015967 |
| Queries with no relevant evidence retrieved | 28 | 28 |
| Zero-result queries | 0 | 0 |
| Queries without positive labels | 0 | 0 |
| Reference evaluator maximum absolute difference | 0.0 | 0.0 |

Both runs produced byte-identical `queries.json` files: every query ID, rank, BM25 score, metric and missed-evidence ID matched. They also produced identical 1,000-draw group-bootstrap intervals: Recall **[0.7186, 0.8790]**, nDCG **[0.5870, 0.7510]**. These 95% intervals are exploratory uncertainty estimates within this development partition, not guarantees of held-out performance.

Reference checks used `pytrec-eval-terrier==0.5.11` (`recall.10` / `ndcg_cut.10`), the same metric interface used by BEIR. Rankings were supplied with order-preserving surrogate scores for tie consistency. The full BEIR package/API was not executed.

## Actual timing and resource results

Each process built the index once, warmed all validation queries once, then measured five seeded shuffled passes: **810 query calls per run**. Retrieval/evaluation/profiling/split code hashes and experiment configurations matched across runs. The report-view histogram formatting and explicit UI dependency declaration were refined between runs; they are outside the measured CLI search path.

| Measure | First run | Reproduction |
| --- | ---: | ---: |
| Query p50 | 18.77 ms | 8.12 ms |
| Query p95 | 36.61 ms | 15.39 ms |
| Query-stage process CPU time | 15.766 s | 6.859 s |
| Query-stage wall time | 15.868 s | 6.908 s |
| Sampled peak process RSS during measured queries | 96.08 MiB | 95.87 MiB |
| Index-build wall time | 0.716 s | 0.671 s |

**Timing varied by more than 2×.** Background load, scheduling, power/frequency state and OS caches were not controlled, and the cause of this difference was not isolated. Retain both runs; do not select the faster timing as a stable product claim. Later model comparisons should use a more controlled/interleaved measurement session.

Per-call latency excludes loading/index construction, metrics/reference evaluation, reports and UI. Stage CPU/RSS includes the runner and sampler. RSS covers the whole process, not isolated index memory; requested 10 ms sampling can miss peaks and be delayed by scheduling. No GPU was used. Read `measurement_protocol` in the manifest before comparing these values with another experiment.

## A concrete failure to inspect

Validation query `0`, “0-dimensional biomaterials lack inductive properties.”, missed judged document `31715818`, “New opportunities: the use of nanotechnologies to manipulate and track stem cells.” Its top result was instead “High-performance neuroprosthetic control by an individual with tetraplegia.”

An explicit token-overlap check found that the missed judged document shares **none** of the query's tokens under the saved tokenizer. The top result shares only `0` and `dimensional`. Because this baseline returns only token-matching candidates, the judged document cannot be retrieved for this query. This explains this particular lexical failure; it does not diagnose all 28 failures or prove that a dense model will solve them. No tokenizer/parameter change was selected from the observation. Unjudged top results are not proven irrelevant.

## Artifacts

Local directories `runs/bm25-validation-v1/` and `runs/bm25-validation-reproduction/` each contain the manifest, exact split, complete query rankings/metrics, raw timing CSV and a concise report. They are included in the milestone 2 ZIP for inspection and remain ignored by Git. The corpus still downloads separately.

Reproduce with the committed configs and a new output directory:

```bash
python -m retrieval_lab.cli verify-split
python -m retrieval_lab.cli evaluate --output runs/my-bm25-validation-001
```

Quality/rankings should match under identical inputs and code. Timings need not match. Official test labels remain reserved for a future frozen-system evaluation.

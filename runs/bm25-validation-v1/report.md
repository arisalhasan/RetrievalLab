# BM25 validation experiment

These are measured train-derived validation results, not official SciFact test scores.

- Validation queries: 162
- Split SHA-256: `091e0d820c4dac0fd0ea80a133773da72984b7890ef15a66b053c8338d8b34c6`
- Recall@10: 0.801337
- nDCG@10: 0.667302
- Query p50 / p95: 18.772 / 36.611 ms
- Measured query calls: 810
- Sampled peak process RSS during queries: 96.08 MiB
- Query-stage process CPU seconds: 15.766
- Queries missing all relevant evidence: 28

See manifest.json for confidence intervals, hardware, versions and measurement caveats; queries.json for rankings/missed evidence; timings.csv for raw measured calls. Unjudged documents are not proven irrelevant.

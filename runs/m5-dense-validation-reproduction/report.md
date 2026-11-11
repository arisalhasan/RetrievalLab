# Dense (MiniLM) validation experiment

These are measured train-derived validation results, not official SciFact test scores.

- Validation queries: 162
- Split SHA-256: `091e0d820c4dac0fd0ea80a133773da72984b7890ef15a66b053c8338d8b34c6`
- Recall@10: 0.811831
- nDCG@10: 0.648212
- Query p50 / p95: 18.823 / 24.137 ms
- Measured query calls: 160
- Distinct timing queries: 32; quality queries: 162
- Sampled peak process RSS during queries: 504.35 MiB
- Query-stage process CPU seconds: 3.125
- Queries missing all relevant evidence: 29

See manifest.json for confidence intervals, hardware, versions and measurement caveats; queries.json for rankings/missed evidence; timings.csv for raw measured calls. Unjudged documents are not proven irrelevant.

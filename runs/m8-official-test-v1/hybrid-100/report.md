# hybrid-100 official held-out test experiment

These are measured official SciFact test scores from settings frozen before test-label access.

- Official test queries: 300
- Split SHA-256: `17e762d1154f7b1415be9a155d9ba31ac60de1553e3a84df9cee081efaa4ef6f`
- Recall@10: 0.810556
- nDCG@10: 0.683420
- Query p50 / p95: 59.109 / 89.415 ms
- Measured query calls: 48
- Distinct timing queries: 16; quality queries: 300
- Sampled peak process RSS during queries: 558.02 MiB
- Query-stage process CPU seconds: 2.984
- Queries missing all relevant evidence: 52

See manifest.json for confidence intervals, hardware, versions and measurement caveats; queries.json for rankings/missed evidence; timings.csv for raw measured calls. Unjudged documents are not proven irrelevant.

# Milestone 2: measure and explain

The first milestone proved the search path works. This milestone asks how much judged evidence it finds, how well it orders it, and what computation costs. BM25's scoring formula and default parameters are unchanged.

## A saved split, not an improvised sample

SciFact has official train and test judgments but no official validation split. We derive **647 training / 162 validation queries** from its 809 official train queries. Related queries are kept together when they share judged evidence or at least 80% token-set overlap. Relationships are transitive: if A is related to B and B to C, all three remain together.

For word sets A and B, **Jaccard similarity = shared words / distinct words across both sets**. If A={cell, growth, increases} and B={cell, growth, decreases}, similarity is 2/4=0.5. This also illustrates a limitation: word similarity cannot guarantee semantic equivalence. Shared positive-evidence grouping catches some variants that token similarity misses; the heuristic is not a full semantic/entity audit.

Seeded hash ordering assigns entire groups, targeting 20% validation. The manifest records exact IDs, groups, data hashes, rules and seed. `verify-split` regenerates the manifest and rejects drift. No positive-evidence document is shared across the two partitions. Indexing the full corpus is still valid: document availability is different from using held-out labels for training or tuning.

## Recall@10: evidence coverage

Suppose a claim has three judged relevant documents. Retrieving two in the top ten gives Recall@10 = 2/3. Finding neither gives zero. We compute this separately for each validation query, then average the query scores so a query with many judgments does not dominate the result.

The first run's **0.8013** means the query-average fraction of judged relevant evidence retrieved in the first ten results was about 80.1%. It does not mean exactly 80.1% of queries succeeded, and it is not precision or factual accuracy.

## nDCG@10: evidence order

Rank 1 contributes relevance/log2(2), rank 2 contributes relevance/log2(3), and rank 3 contributes relevance/log2(4). Higher positions get more credit. Divide the ranking's discounted gain by the best possible discounted gain for that query; an ideal ranking scores 1.

For one relevant document returned at rank 2, nDCG is 1/log2(3), approximately 0.6309. At rank 1 it is 1. If it is outside the cutoff it contributes zero. With multiple relevant documents, the ideal denominator includes them up to the cutoff.

The code uses linear grades, following TREC's definition; SciFact's grades are binary. Queries without results or positive labels stay in our macro average with zero. Unjudged documents contribute zero evaluation gain but are not proven irrelevant. Tests compare hand calculations and reference-evaluator outputs, including short rankings and self IDs.

## What “fast” and “small” mean here

The runner builds an index once, warms up all validation queries, then makes five shuffled measured passes. Query latency excludes indexing, data loading, evaluation and page rendering. p50 is the middle of the observed latency distribution; p95 is the interpolated 95th percentile, so roughly 95% of measured calls are no slower than it.

The memory figure is **sampled peak process RSS**: resident memory across the entire process, including Python and corpus objects. It is not the model size or isolated index allocation. A requested 10 ms sample period can miss spikes or be delayed by scheduling. CPU seconds count actual process CPU time during the stage, including the runner and sampler; they differ from wall time.

Local machine load and OS caches are uncontrolled. Reproduction should preserve quality/rankings; it should not be expected to preserve exact timings. Read the saved protocol before comparing another system's speed.

## Code reading order

1. `splits.py`: follow how connected groups become saved query IDs.
2. `metrics.py`: work through the Recall/nDCG equations and reference check.
3. `profiling.py`: separate the search-call timer from whole-process resource sampling.
4. `evaluation.py`: trace warmup, measured passes, scoring, and saved artifacts.
5. `dashboard.py`: compare a failed query's missed evidence and retrieved ranking without recomputing the experiment.

The CPU baseline requires no training and no GPU. No model selection using test judgments has occurred.

One observed failure makes the next milestone concrete: query `0` has no token overlap with its judged relevant document under the current tokenizer. Its top result shares only `0` and `dimensional`. BM25 cannot retrieve the judged document through shared words in this case. Dense retrieval will test whether semantic representations help, using the same validation partition.

## Student verification checkpoint

From the project root, using your virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe -m retrieval_lab.cli verify-split
.\.venv\Scripts\python.exe -m retrieval_lab.cli evaluate --output runs/my-bm25-validation-001
$env:RUN_SCIFACT_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use an unused output directory. In **Validation & failures**, check that the saved run shows 162 validation queries, Recall@10 about 0.8013 and nDCG@10 about 0.6673 under the committed config. Your timing/resource numbers may differ. Inspect one query with missed evidence and explain why exact word matching might have failed; treat any explanation as a hypothesis until you inspect the documents.

Share the test summary and one failure observation before milestone 3. The [execution record](VERIFICATION_MILESTONE_2.md) separates the agent's checks from tasks you still need to verify locally.

## Three interview questions

1. Why keep related claims in the same partition instead of randomly splitting individual queries?
2. How can two systems have the same Recall@10 but different nDCG@10?
3. Why measure warmed query latency separately from indexing, and what does sampled process RSS fail to tell you?

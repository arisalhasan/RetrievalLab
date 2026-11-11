# Experiment policy

Milestone 1 single-query records are smoke checks. Milestone 2 produces measured train-derived validation runs. Neither is an official held-out SciFact test result.

## Dataset roles

Official BEIR SciFact provides train and test judgments, with no official dev split. Preserve the source files and name derived splits clearly, for example `train-derived-validation-v1`.

The committed manifest `configs/splits/scifact-train-v1.json` freezes 647 training and 162 validation query IDs (seed 42) before the milestone 2 scores were computed. Grouping joins queries sharing a positive-evidence document or token-set Jaccard similarity ≥0.8, then takes connected components so transitive relationships remain together. Deterministic hash ordering chooses a prefix nearest the target 20% validation size while leaving both partitions nonempty. The 490 groups have maximum size 8; 98 groups are in validation. Positive-evidence document overlap is zero.

Group membership, settings, data hashes, exact IDs, and the canonical manifest checksum are persisted. Loading regenerates and verifies the entire manifest. This heuristic reduces obvious overlap; it does not guarantee semantic, entity, or topic independence. Validation is development data and some official train claims were already explored in milestone 1. Official test labels were reserved through development; milestone 8 opens them only after the final release freeze.

Fit learned components and mine training negatives only from the derived training partition. Validation chooses parameters, candidates, epochs, and checkpoints; do not use its labels for gradient updates. Do not change split seeds/rules after seeing quality without declaring a separately versioned experiment. No BM25 parameter tuning was done for the first validation run.

Indexing the full shared corpus without labels is allowed: retrieval needs all candidates. Test judgments must not choose tokenizer settings, k1/b, encoder, fusion settings, reranker, depth, or checkpoints. Final test error analysis happens after freezing systems. A system tuned after test inspection cannot make the same untouched-holdout claim.

For neural models, record model ID plus immutable revision, tokenization/truncation, precision, device, normalization, pooling, and any upstream SciFact exposure disclosed in its model card. Pretraining may include benchmark exposure; do not automatically claim zero-shot generalization.

## Metric definitions

- **Recall@10:** judged relevant documents in the first ten results divided by all judged relevant documents for that query.
- **nDCG@10:** discounted gain of the first ten results divided by the best possible discounted gain. Relevant results near the top contribute more.
- **Aggregation:** query macro-average over every query in the selected partition. Queries with no positive labels or no results receive zero and remain in the average. No-positive counts are reported explicitly. A runtime failure aborts the run; it is not silently removed.
- **Compatibility:** nDCG uses linear relevance grades, matching TREC's `ndcg_cut`; with binary SciFact labels this equals exponential binary gain. Query/document self IDs are removed after retrieval without candidate backfill. The denominator still uses all positive judgments, matching BEIR's default filtering behavior.
- **Ties:** BM25 preserves score-descending, string-document-ID-ascending order. Reference checks use rank-preserving surrogate scores so pytrec_eval sees the same ordering. Raw BM25 scores remain in artifacts. This does not assert identical tie policy for an independently submitted raw-score run in every evaluator.

The runner checks every per-query metric against pytrec_eval's `recall.10` and `ndcg_cut.10`, supplied by `pytrec-eval-terrier==0.5.11`. This is the backend interface used by [BEIR's evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py); the full BEIR package/API was not run in milestone 2. [TREC documents linear gains in its implementation](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c).

Quality intervals use 1,000 seeded cluster-bootstrap draws of whole validation groups. Each draw computes a query-weighted macro mean; related claims never resample independently. Percentile limits are 2.5% and 97.5%. These exploratory intervals reflect uncertainty within this small development partition, not population-wide guarantees or proof of improvement over another model.

Use identical query IDs, corpus, cutoff, candidate constraints, and metric semantics across systems. Save per-query rankings/metrics. Unjudged is not a proven negative; training negative mining needs special care.

## Timing and resources

The baseline uses a standalone CPU process: one index build, one full warmup pass, then five independently shuffled passes of the 162 validation queries (810 timed calls; seed 42). Query timers include tokenization, scoring, and top-k selection. They exclude data loading, indexing, metrics/reference evaluation, reports, and UI. Percentiles use linear interpolation over all timed calls. There is one search worker and no query-result cache; OS cache and background workstation load are uncontrolled.

`psutil` measures process CPU seconds and resident memory separately during index construction and measured query passes. A background sampler requests 10 ms intervals and takes start/end samples; actual scheduling may be slower, especially under Python's GIL. Sampled peaks can miss brief peaks. RSS is whole-process memory, including Python and corpus data, not isolated index memory. Stage CPU/wall measurements include runner/sampler overhead. One-core-equivalent CPU percentages may exceed 100% and are not the same as Task Manager's machine-wide percentage. No GPU is used. Data loading and warmup resources are not separately profiled yet.

## Run artifacts

Each evaluation directory is created atomically and never overwrites an earlier run. It contains a manifest (metrics/intervals/protocol, hardware, full installed-package snapshot, source/config/data hashes, Git state when available), the exact split manifest, per-query rankings/metrics/missed IDs, raw timing samples, and a Markdown report. Artifact hashes and recalculated aggregates are checked by the dashboard. This protects against accidental edits, not malicious fabrication of an entirely new self-consistent report.

`requirements.lock.txt` pins the reference environment. Generated run directories remain ignored by Git; reviewed summaries live in `docs/results/`. Keep the first baseline even when reproducing it. No experiment tracking server is required.

## Milestone 3 protocol

The pretrained encoder and all settings in `configs/dense.json` were fixed before observing this milestone's validation scores. The immutable model revision, weight hashes, mean pooling, 256-token right truncation, title/newline/abstract input, normalized float32 vectors, eager CPU execution, seed 42, and single PyTorch/FAISS thread are persisted. No prompts, gradient updates or label-driven selection occur. Upstream pretraining overlap remains unaudited.

FAISS exact inner product is checked against independent NumPy scores on every validation query vector. Full-corpus score sorting makes tied cutoffs deterministic. The oracle records float32 near-tie differences within 1e-6 rather than treating those as substantive errors. scikit-learn provides another independent synthetic cosine check. Metric reference checks continue through `pytrec_eval`; the full BEIR neural API is not executed.

Fresh BM25 and dense runs use the same frozen split and evaluation config under the milestone 3 environment. Warm query time includes fresh query encoding for dense and the entire retrieval call, with no query cache. Preparation records framework/model loading and either original document encoding or verified cache reuse. Corpus truncation counts and original encoding cost survive cache reuse. RSS remains whole-process sampled memory, not isolated vector-index memory.

Paired candidate-minus-baseline differences and intervals resample whole frozen validation groups. The comparison rejects mismatched data/splits/metric definitions/evaluation settings. Workstation timing remains uncontrolled; repeats expose variability, and all runs are retained. The CPU reference lock is `requirements-dense.lock.txt`; the lean base lock remains available for BM25. None of these development results is an official test score.

## Milestone 4 protocol

The primary hybrid is equal-weight RRF with constant 60 and up to 100 candidates per source. These settings were fixed before milestone 4 measurements, after the previously reported milestone 3 development analysis. One depth-10 ablation was also declared before these measurements; it changes only candidate depth. Neither is presented as a fresh validation holdout or a test-selected configuration. No rank-constant sweep or automatic best-config selection is implemented/executed.

The exact component configs are embedded in the hybrid config and copied for the standalone BM25/dense baselines. The actual experiment script records their snapshots, runs each system sequentially in its own process, repeats each, and makes same-split paired comparisons. Historical milestone 2/3 runs remain immutable. The dense encoder/vector files and BM25 scoring code remain unchanged; valid corpus embeddings are reused.

Fusion uses one-based source-list positions; absent documents contribute zero; scores are summed accurately, then ties use ascending string document IDs. Duplicate IDs in a source and conflicting document content across sources abort. Both sources are required: there is no silent single-method fallback. Query timing includes both live searches at configured depth, query encoding, RRF and trace creation. Candidate diagnostics are computed in an additional untimed pass and labels never enter either component or fusion.

Candidate recall is evaluated on the already truncated source lists/union, after excluding self IDs. It is a coverage ceiling rather than final Recall@10; relevant candidates can still be lost through ordering. The report distinguishes candidate misses from fusion losses, retains exact source IDs/contributions, and verifies those derived values on load. Interpret this with incomplete relevance judgments and existing grouping/pretraining/truncation limits.

## Milestone 5 protocol

Before milestone 5 results, the pretrained baseline was fixed to `cross-encoder/ms-marco-TinyBERT-L2-v2`, revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`, raw identity-activated scalar logits, CPU float32 eager attention, batch 16, one thread, seed 42. It reranks the top 50 documents from the unchanged primary hybrid (100 candidates per child, RRF constant 60). The joint query/title/abstract input is limited to 512 tokens with longest-first truncation. Model selection was for a compact CPU baseline, not a SciFact validation sweep. No fitting, negatives, gradient updates or test-label selection occur. Upstream exposure remains unaudited.

Every one of the 162 validation queries contributes to quality. To bound CPU experiment cost, the new explicit profile selects 32 validation IDs by uniform seed-42 sampling without using relevance labels to choose timing cases. An untimed full-quality pass precedes a warm-up on those IDs and five shuffled measured passes (160 calls). The same IDs/profile apply to all four fresh baselines/reranked systems and repeats. Sampled timing does not represent all queries or production load. Older 810-call results remain separate; omit/null `latency_query_limit` to use all-query timing in a new experiment.

Latency includes fresh hybrid retrieval/encoding, joint scoring of each actual candidate pair, sorting and trace creation. Input token-length/coverage diagnostics are untimed. Quality traces are retained solely for later analysis, never used to bypass timed inference. Tests verify live repeated scorer calls and full quality coverage with a smaller timing sample. Reports validate the exact timing IDs/repetitions, and comparisons reject mismatched timing samples/configs.

Rerank input recall is measured on the actual hybrid top 50, not its larger BM25/dense union. Preserve candidate omissions, final ordering losses, pre-rerank and post-rerank metrics, every pair's raw score, model hashes and pair truncation counts. Results may regress; raw logits are uncalibrated relevance scores, not scientific-truth probabilities. The new model and larger pair budget are both changed factors, so the comparison cannot isolate only the architectural effect of cross-attention.

## Training and final test protocol

The predeclared two-epoch training recipe uses only the frozen 647 training query IDs. Save every positive and assumed-negative ID, negative source, seed, parent revision, train loss, validation rankings, reference metric check and checkpoint SHA. Epoch 0 is eligible. Select highest full-validation nDCG@10, then Recall@10, then earliest epoch. A training loss reduction does not establish retrieval improvement. Keep the best trained checkpoint as an experimental comparison even if the original model wins.

The release candidate rule uses highest validation nDCG@10, then Recall@10, then name. Configurations and verified development evidence, source/model/data hashes and runtime are locked in `releases/v1/freeze.json` before official test-label parsing. Final results cannot change this selection. Test-label access is explicit and logged. Do not change these choices after test inspection or present another test pass as a fresh untouched holdout.

The release timing profile is intentionally separate from older experiments: all official test queries for quality; the same seed-42 label-independent sample of 16 IDs for each system's one warmup and three measured passes (48 timing calls). Compare only identical profiles. Test uncertainty uses query bootstrap with 1,000 samples; test claims are singleton groups, so dependence can understate uncertainty. All quality metrics retain the same definitions and pytrec_eval checks. Final figures separate quality, timing and memory scales. See [milestone 8](MILESTONE_8.md) for reproduction and execution limits.

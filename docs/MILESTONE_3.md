# Milestone 3: search by vector similarity

RetrievalLab now has two ways to find evidence. BM25 rewards shared words. The new dense retriever uses a pretrained neural network to represent the meaning of each title/abstract and query as 384 numbers. These representations are called embeddings. Similar directions suggest related text; the search returns the closest abstracts. Neither system decides whether a scientific claim is true.

## Run it locally

Use Python 3.12 and the existing virtual environment. First complete the README's base setup and download SciFact. On Windows x86-64, add the CPU dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dense.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev,dense]"
.\.venv\Scripts\python.exe -m retrieval_lab.cli model-download
.\.venv\Scripts\python.exe -m retrieval_lab.cli build-dense
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The reference lock includes the official Windows Python 3.12 CPU PyTorch wheel. For other platforms, install `python -m pip install -e ".[dev,dense]"` using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for a suitable CPU build. Those platform installs have not been executed here. Allow several GB for Python dependencies, approximately 91 MB for model weights, and about 8 MB for document vectors. Corpus encoding runs once on the CPU; time depends on your laptop. No GPU, account or paid service is needed.

Select **Dense** in the sidebar, enter a query, and select **Search**. The model and document cache must already exist: the dashboard gives preparation instructions instead of downloading or encoding the corpus during a search. After preparation, inference works offline. The first switch to Dense loads the model and may take longer than later interactions.

Run both methods on the same saved validation partition:

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone3.py --name student-m3-v1
```

This verifies the split, verifies/downloads the pinned model, prepares/reuses document embeddings, runs BM25 and dense sequentially in separate processes, and saves a paired comparison. Use a new name for each experiment. Reload the dashboard, open **Compare systems**, choose the two runs, and select **Compare runs**. Use **Validation & failures** to inspect each method's missed judged evidence and full retrieved ranking.

Equivalent individual commands:

```bash
python -m retrieval_lab.cli evaluate --system bm25 --output runs/my-m3-bm25
python -m retrieval_lab.cli evaluate --system dense --output runs/my-m3-dense
python -m retrieval_lab.cli compare --baseline runs/my-m3-bm25 --candidate runs/my-m3-dense --output comparisons/my-m3.json
```

## Follow one query through the code

1. `app.py` handles the form and chooses a retriever. `retrievers.py` gives the evaluator the same `search(query, top_k)` interface for both methods.
2. `encoders.py` loads verified local model files using Sentence Transformers and PyTorch. It produces normalized float32 vectors on the CPU. It never receives relevance labels.
3. `dense.py` encodes title + newline + abstract once per document and stores the vectors in a checked cache. It encodes the query afresh for every search call.
4. `vectors.py` searches a FAISS `IndexFlatIP`. The result objects have the same document/score shape as BM25, so the shared UI and evaluator can consume them.
5. `evaluation.py` checks exact search against NumPy, runs warm-up and timed passes, then evaluates the rankings with official training relevance labels for the frozen validation IDs. `comparison.py` pairs the two methods' scores query by query.

`dense_config.py` and `configs/dense.json` declare all settings. `model_store.py` downloads only known files at an immutable model commit, verifies upstream blob hashes and a known weights SHA-256, and installs atomically. It uses TLS verification and bounded byte-range requests for weights. It never executes downloaded Python code. `files.py` provides bounded retries for observed temporary Windows file locks without overwriting existing evidence.

## The mathematics, without a neural-network course

An embedding is a point in a space with 384 axes. You cannot reliably name what each individual axis means; the model learned a distributed representation from its training examples.

For vectors **q** and **d**, cosine similarity is:

`cosine(q, d) = dot(q, d) / (length(q) * length(d))`

Normalization divides each vector by its length, making that length 1. The formula then becomes a dot product: multiply matching coordinates and add them up. FAISS performs these comparisons efficiently. Its [metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) explains the normalized inner-product relationship. Similarity is a retrieval score, not a probability, evidence judgment or truth estimate.

`IndexFlatIP` examines every document vector, so it is exact rather than approximate. For this 5,183-document corpus we request every score and sort by descending score, then ascending document ID for ties. This handles ties even at the top-k boundary deterministically. It costs O(N × dimension) for scoring and O(N log N) for sorting, with O(N) scores returned. This deliberately simple implementation would need a different candidate/tie strategy at millions of documents. NumPy independently verifies the same vectors; tiny float32 differences within 1e-6 can reorder near ties and are disclosed in the check report.

## Fixed choices and important limits

- Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. 384 dimensions, attention-mask mean pooling, unit normalization, float32 CPU, eager attention, batch size 32. Seeds and PyTorch/FAISS CPU threads are fixed at 42 and 1 respectively.
- Inputs are truncated on the right to **256 model tokens, including special tokens**. A token is not necessarily a whole word. Titles consume part of the budget. Cache/report metadata counts truncation, so this limitation remains visible. Full abstracts are still displayed in the UI.
- Model selection and these settings were declared before this milestone's validation run. There is no fine-tuning, test-label evaluation, or validation-driven hyperparameter sweep in this milestone. Upstream pretraining overlap with SciFact has not been audited; do not claim a strict zero-shot novelty result.
- Cache identity includes document content/order, model file hashes, runtime versions, configuration and embedding implementation hashes. Files are checked for integrity, vector dimensions, unit norms and document alignment. Changing an identity creates a new cache. Corruption raises an error; it is never silently accepted. Use a new cache root if an existing cache is damaged.
- BM25 and cosine scores have different scales. Compare ranked results and metrics, not their raw score values. A pretrained dense model can lose to lexical matching on specialized scientific terms.
- The 647/162 train/validation partition and seed 42 remain unchanged. Queries with shared positive evidence or near-duplicate wording stay in one group. Labels from the official test set remain reserved. This grouping reduces leakage but does not prove topic independence.

## Read the measurements fairly

Recall@10 asks what fraction of known relevant abstracts appear in the first 10. nDCG@10 also rewards placing evidence earlier. The paired report computes candidate-minus-baseline differences and 95% intervals by resampling whole validation groups. An interval crossing zero does not establish a dependable difference on this sample.

Query timing includes the full `search()` call: fresh query encoding for dense, vector search and canonical sorting. Document/model preparation is separate. One warm-up pass precedes five shuffled measured passes: 810 calls per method. There is no query-result cache. Resource measurements are process CPU time and sampled resident RAM (RSS), not per-model memory, energy or GPU usage. Heavy framework imports can dominate a fresh dense process. Cached preparation is not the original corpus encoding cost; both are recorded.

Runs occur sequentially on an uncontrolled workstation. Keep the original timings and any reproduction timings; do not present the fastest one as a stable guarantee. See the measured report and execution record for what actually ran.

The [actual milestone 3 report](results/DENSE_COMPARISON_V1.md) shows higher dense recall but lower nDCG, with both paired uncertainty intervals crossing zero. It also records 7 minutes 46 seconds of original document encoding and the substantial document-truncation count. The [execution record](VERIFICATION_MILESTONE_3.md) distinguishes real benchmark/model checks from synthetic tests and unexecuted platforms/features.

## Tests and your checkpoint

Default tests use explicitly synthetic vectors where they substitute for a neural model. They never become SciFact results and never download anything. FAISS output is also compared with independent scikit-learn cosine calculations. The real-model test is opt-in:

```powershell
$env:RUN_SCIFACT_TESTS = '1'
$env:RUN_DENSE_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Please verify locally: search the same claim with BM25 and Dense; open both saved validation reports; inspect one query where Dense improves and one where it gets worse. Explain the tradeoff using the abstracts and labels rather than assuming the neural method is better.

Three internship interview questions:

1. What is an embedding, and why does normalizing vectors make inner product equal cosine similarity?
2. Why can a pretrained dense retriever lose to BM25, and how would you investigate a failed scientific query?
3. Why must dense query latency include encoding while corpus encoding is reported separately, and how do we avoid using test labels for model selection?

The next milestone is reciprocal rank fusion: combine BM25 and dense rankings while respecting their different score scales. No hybrid retrieval or reranker training is implemented here.
